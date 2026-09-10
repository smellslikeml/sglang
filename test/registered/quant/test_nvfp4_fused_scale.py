#!/usr/bin/env python3

import unittest

import torch

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.srt.layers.quantization.nvfp4_fused_scale import (
    needs_fused_nvfp4_reconcile,
    reconcile_fused_nvfp4_block_scales,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GROUP_SIZE = 16


def _int_block_scales(rows: int, cols: int) -> torch.Tensor:
    """Block scales drawn from 1..8 — all exactly representable in FP8-E4M3, so
    the reconcile math can be checked for *exact* equality when the global-scale
    ratios are powers of two."""
    generator = torch.Generator().manual_seed(0)
    ints = torch.randint(1, 9, (rows, cols), dtype=torch.int32, generator=generator)
    return ints.to(torch.float8_e4m3fn)


def _build_fused_layer(
    method: ModelOptFp4LinearMethod,
    logical_widths,
    hidden_size: int,
    global_scales,
) -> torch.nn.Module:
    """Materialize a fused NVFP4 linear through create_weights and fill it as a
    per-module-calibrated checkpoint would (distinct weight_scale_2 per module)."""
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=hidden_size,
        output_partition_sizes=list(logical_widths),
        input_size=hidden_size,
        output_size=sum(logical_widths),
        params_dtype=torch.bfloat16,
    )
    out_rows = sum(logical_widths)
    layer.weight_scale.data.copy_(
        _int_block_scales(out_rows, hidden_size // GROUP_SIZE)
    )
    layer.weight_scale_2.data.copy_(torch.tensor(global_scales, dtype=torch.float32))
    return layer


class TestNvFp4FusedScale(CustomTestCase):
    def setUp(self):
        self.method = ModelOptFp4LinearMethod(
            ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True, group_size=GROUP_SIZE
            )
        )

    def test_reconcile_preserves_effective_scale(self):
        # Derived property: folding per-module global scales into the block
        # scales, then serving with the common max global scale, must reproduce
        # each module's original effective scale block-for-block.
        logical_widths = [4, 4]
        cols = 3
        block = _int_block_scales(sum(logical_widths), cols)
        global_scales = torch.tensor([1.0, 0.5])  # power-of-two ratio -> exact

        reconciled = reconcile_fused_nvfp4_block_scales(
            weight_scale=block,
            weight_scale_2=global_scales,
            logical_widths=logical_widths,
        )
        common = global_scales.max()

        start = 0
        for idx, width in enumerate(logical_widths):
            end = start + width
            original_effective = block[start:end].float() * global_scales[idx]
            served_effective = reconciled[start:end].float() * common
            torch.testing.assert_close(
                served_effective, original_effective, rtol=0, atol=0
            )
            start = end

    def test_method_reconciles_fused_layer(self):
        # Wiring: the seam process_weights_after_loading invokes must rescale the
        # smaller-global-scale module's block scales down by its global ratio and
        # leave the argmax module untouched.
        logical_widths = [8, 8]
        hidden = 32
        layer = _build_fused_layer(
            self.method, logical_widths, hidden, global_scales=[1.0, 0.5]
        )
        before = layer.weight_scale.data.float().clone()

        self.method._reconcile_fused_global_scale(layer)
        after = layer.weight_scale.data.float()

        # Module 0 holds the common (max) global scale -> unchanged.
        torch.testing.assert_close(after[:8], before[:8], rtol=0, atol=0)
        # Module 1 was calibrated at 0.5x -> block scales halved.
        torch.testing.assert_close(after[8:], before[8:] * 0.5, rtol=0, atol=0)

    def test_method_noop_on_uniform_global_scale(self):
        # Negative branch: when every fused module shares one global scale there
        # is nothing to repair; the predicate must not degrade to "always
        # reconcile" and inject FP8 rounding into the common case.
        logical_widths = [8, 8]
        hidden = 32
        layer = _build_fused_layer(
            self.method, logical_widths, hidden, global_scales=[0.25, 0.25]
        )
        self.assertFalse(needs_fused_nvfp4_reconcile(layer.weight_scale_2))

        before = layer.weight_scale.data.float().clone()
        self.method._reconcile_fused_global_scale(layer)
        torch.testing.assert_close(
            layer.weight_scale.data.float(), before, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
