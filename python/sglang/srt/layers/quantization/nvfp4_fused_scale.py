# SPDX-License-Identifier: Apache-2.0

"""Fused-GEMM global-scale reconciliation for NVFP4 (W4A4) checkpoints.

NVFP4 uses two levels of weight scaling: a per-16-element block scale stored in
FP8-E4M3 (``weight_scale``) and one FP32 per-module global scale
(``weight_scale_2``). The dequantized weight of an element is::

    w = fp4_code_value * float(weight_scale[row, block]) * weight_scale_2[module]

ModelOpt calibrates ``weight_scale_2`` per module. SGLang, however, fuses
several such modules into a single GEMM (e.g. the packed QKV projection, or
Qwen3's Gated DeltaNet ``in_proj_qkvz`` / ``in_proj_ba``). The fused native
kernel consumes a *single* scalar global scale — in practice ``max`` over the
per-module ``weight_scale_2`` — while the per-block scales it multiplies were
calibrated against each module's *own* global scale. When the per-module global
scales differ, every block outside the ``argmax`` module is dequantized too
large by ``max(weight_scale_2) / weight_scale_2[module]``, a systematic bias
that no amount of activation quantization corrects.

This module repairs that mismatch the same way the FP8 path's
``requantize_with_max_scale`` does, but for the NVFP4 two-level scale: it folds
each module's global scale into its block scales so that serving with one
common global scale preserves the original effective per-block scale.

Adapted from "Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for
the Recurrent Half of a Hybrid 27B LLM" (arXiv:2609.04098), which identifies and
repairs this global-scale mismatch for per-module-calibrated NVFP4 checkpoints
served by module-fusing kernels.
"""

from typing import List

import torch

# Ratios within this tolerance of 1.0 are treated as "already common": the
# modules share a global scale, so reconciliation would only add FP8 rounding
# noise. Chosen well below FP8-E4M3's coarsest relative step so genuine
# mismatches are never skipped.
_RECONCILE_RATIO_EPS = 1e-6


def needs_fused_nvfp4_reconcile(weight_scale_2: torch.Tensor) -> bool:
    """Return True iff a fused NVFP4 layer packs modules with materially
    different per-module global scales.

    False for single-module layers (nothing to fuse), for already-common
    global scales (unfused checkpoints broadcast one value across the slots),
    and for uninitialized / non-positive scales (leave those untouched).
    """
    if weight_scale_2 is None or weight_scale_2.numel() < 2:
        return False
    vals = weight_scale_2.detach().float().reshape(-1)
    if not torch.isfinite(vals).all() or bool((vals <= 0).any()):
        return False
    return bool((vals.max() / vals.min()) > 1.0 + _RECONCILE_RATIO_EPS)


def reconcile_fused_nvfp4_block_scales(
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    logical_widths: List[int],
) -> torch.Tensor:
    """Fold per-module global scales into the FP8 block scales.

    Returns a new ``weight_scale`` (FP8-E4M3, same shape) such that dequantizing
    with the single common global scale ``max(weight_scale_2)`` reproduces each
    module's original effective per-block scale
    ``weight_scale[row, block] * weight_scale_2[module]``.

    ``weight_scale`` is row-partitioned by ``logical_widths`` (one contiguous
    row block per fused module), matching ``create_weights`` layout.
    """
    if sum(logical_widths) != weight_scale.shape[0]:
        raise ValueError(
            f"logical_widths sum {sum(logical_widths)} does not match "
            f"weight_scale rows {weight_scale.shape[0]}"
        )
    if len(logical_widths) != weight_scale_2.numel():
        raise ValueError(
            f"logical_widths count {len(logical_widths)} does not match "
            f"weight_scale_2 entries {weight_scale_2.numel()}"
        )

    globals_f = weight_scale_2.detach().float().reshape(-1)
    common = globals_f.max()

    # Work in FP32 to avoid compounding FP8 rounding across the rescale, then
    # cast back once at the end.
    reconciled = weight_scale.detach().float()
    start = 0
    for idx, width in enumerate(logical_widths):
        end = start + width
        # ratio <= 1, so block scales only shrink and stay representable.
        ratio = globals_f[idx] / common
        reconciled[start:end, :] = reconciled[start:end, :] * ratio
        start = end

    return reconciled.to(weight_scale.dtype)
