# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the lattice vector-quantized linear method (leech_vq).

Adapted from arXiv:2609.02652 ("Unfolding the Leech Lattice"). Exercises the
serving path wired into the quantization registry: codebook-indexed storage ->
offline expansion (``process_weights_after_loading``) -> dequant-plus-GEMV
(``apply``).
"""

import unittest

import torch

from sglang.srt.layers.quantization import (
    BASE_QUANTIZATION_METHODS,
    get_quantization_config,
)
from sglang.srt.layers.quantization.leech_vq import (
    LeechVQConfig,
    LeechVQLinearMethod,
    bits_per_weight,
    dequant_vector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLeechVQRegistration(CustomTestCase):
    def test_registered_in_quantization_methods(self):
        # Completeness contract: the wiring edit must expose leech_vq through
        # the shared registry, not just as an importable class.
        self.assertIn("leech_vq", BASE_QUANTIZATION_METHODS)
        self.assertIs(get_quantization_config("leech_vq"), LeechVQConfig)


class TestLeechVQDecode(CustomTestCase):
    def _build_layer(self, config, out_features, in_features, params_dtype):
        method = LeechVQLinearMethod(config)
        layer = torch.nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=in_features,
            output_partition_sizes=[out_features],
            input_size=in_features,
            output_size=out_features,
            params_dtype=params_dtype,
        )
        return method, layer

    def test_offline_expand_matches_reference_gemv(self):
        # Derived property: expansion then apply must equal a hand-rolled
        # gather -> scale -> reshape -> F.linear over the same codebook.
        torch.manual_seed(0)
        vec_dim, codebook_size = 4, 16
        out_features, in_features = 8, 12  # in_features divisible by vec_dim
        n_groups = in_features // vec_dim
        dtype = torch.float32

        config = LeechVQConfig(vec_dim=vec_dim, codebook_size=codebook_size)
        method, layer = self._build_layer(config, out_features, in_features, dtype)

        codebook = torch.randn(codebook_size, vec_dim, dtype=dtype)
        qindices = torch.randint(0, codebook_size, (out_features, n_groups))
        weight_scale = torch.rand(out_features, 1, dtype=dtype) + 0.5
        layer.codebook.data.copy_(codebook)
        layer.qindices.data.copy_(qindices.to(torch.int32))
        layer.weight_scale.data.copy_(weight_scale)

        method.process_weights_after_loading(layer)

        # The compact source layout is freed; only the dense weight remains.
        self.assertTrue(hasattr(layer, "weight"))
        self.assertFalse(hasattr(layer, "qindices"))
        self.assertEqual(tuple(layer.weight.shape), (out_features, in_features))

        x = torch.randn(3, in_features, dtype=dtype)
        out = method.apply(layer, x)

        ref_weight = dequant_vector(qindices, codebook, weight_scale)
        ref_out = torch.nn.functional.linear(x, ref_weight)
        torch.testing.assert_close(out, ref_out)

    def test_expand_without_scale(self):
        # The has_weight_scale=False branch must decode purely from the codebook.
        vec_dim, codebook_size = 2, 8
        out_features, in_features = 4, 6
        n_groups = in_features // vec_dim
        dtype = torch.float32

        config = LeechVQConfig(
            vec_dim=vec_dim, codebook_size=codebook_size, has_weight_scale=False
        )
        method, layer = self._build_layer(config, out_features, in_features, dtype)
        self.assertFalse(hasattr(layer, "weight_scale"))

        codebook = torch.randn(codebook_size, vec_dim, dtype=dtype)
        qindices = torch.randint(0, codebook_size, (out_features, n_groups))
        layer.codebook.data.copy_(codebook)
        layer.qindices.data.copy_(qindices.to(torch.int32))

        method.process_weights_after_loading(layer)
        ref_weight = dequant_vector(qindices, codebook, None)
        torch.testing.assert_close(layer.weight, ref_weight)

    def test_create_weights_rejects_indivisible_input(self):
        config = LeechVQConfig(vec_dim=24, codebook_size=256)
        method = LeechVQLinearMethod(config)
        layer = torch.nn.Module()
        with self.assertRaises(ValueError):
            method.create_weights(
                layer,
                input_size_per_partition=100,  # not divisible by 24
                output_partition_sizes=[16],
                input_size=100,
                output_size=16,
                params_dtype=torch.float32,
            )


class TestLeechVQRate(CustomTestCase):
    def test_on_disk_bits_per_weight(self):
        # Derived property: log2(K)/vec_dim. 256-entry codebook over dim-24
        # groups is 8 bits per group => 1/3 bit per weight.
        self.assertAlmostEqual(bits_per_weight(256, 24), 8.0 / 24.0)
        self.assertAlmostEqual(bits_per_weight(16, 4), 1.0)


if __name__ == "__main__":
    unittest.main()
