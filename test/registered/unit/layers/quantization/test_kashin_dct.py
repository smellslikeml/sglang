"""Tests for Kashin-DCT 2-bit-factor weight quantization.

Covers the capability module (transform, factorization, 2-bit codes) and the
registry/linear-method wiring added in
``sglang.srt.layers.quantization.__init__``.
"""

import unittest

import torch

# Non-new module: the quantization registry we wired the new method into.
from sglang.srt.layers.quantization import get_quantization_config
from sglang.srt.layers.quantization.kashin_dct import (
    KashinDCTConfig,
    KashinDCTLinearMethod,
    kashin_dct_quantize,
    orthonormal_dct,
    orthonormal_idct,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _naive_uniform(x: torch.Tensor, bits: int) -> torch.Tensor:
    m = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    levels = 2**bits - 1
    q = torch.round((x / m * 0.5 + 0.5) * levels)
    return (q / levels - 0.5) * 2 * m


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


class TestKashinDCT(unittest.TestCase):
    def test_registered_in_quantization_registry(self):
        # Guards the __init__ wiring: forgetting the registry entry (or renaming
        # get_name) would silently make --quantization kashin_dct unusable.
        self.assertIs(get_quantization_config("kashin_dct"), KashinDCTConfig)
        self.assertEqual(KashinDCTConfig.get_name(), "kashin_dct")
        cfg = KashinDCTConfig.from_config({"num_iters": 10, "kashin_const": 1.1})
        self.assertEqual(cfg.num_iters, 10)
        self.assertAlmostEqual(cfg.kashin_const, 1.1)

    def test_transform_is_orthonormal(self):
        # The sign-randomized DCT must be a true orthogonal transform for the
        # Kashin factorization's adjoint == inverse assumption to hold.
        torch.manual_seed(0)
        x = torch.randn(4, 256)
        self.assertLess(_rel_err(orthonormal_idct(orthonormal_dct(x)), x), 1e-4)
        transformed_norm = orthonormal_dct(x).norm().item()
        self.assertAlmostEqual(transformed_norm, x.norm().item(), places=2)

    def test_codes_are_two_bit(self):
        torch.manual_seed(0)
        q = kashin_dct_quantize(torch.randn(32, 256))
        for codes in (q.codes_a, q.codes_b):
            self.assertEqual(codes.dtype, torch.uint8)
            self.assertTrue(int(codes.max()) <= 3 and int(codes.min()) >= 0)
        # Two 2-bit factors -> ~4 bits/weight (plus small per-row center overhead).
        self.assertLess(q.bits_per_weight(), 5.0)

    def test_two_factor_reconstruction_beats_naive_2bit(self):
        # Core paper claim: the bounded-l-infinity two-factor representation
        # reconstructs far better than a naive 2-bit quantizer and is on par
        # with a naive 4-bit baseline (two 2-bit factors == 4 bits).
        torch.manual_seed(0)
        w = torch.randn(64, 512) * 0.1
        recon = kashin_dct_quantize(w).dequantize(torch.float32)
        kashin_err = _rel_err(recon, w)
        self.assertLess(kashin_err, _rel_err(_naive_uniform(w, 2), w))
        self.assertLess(kashin_err, 1.2 * _rel_err(_naive_uniform(w, 4), w))

    def test_stable_on_outlier_weights(self):
        # Robustness claim: stays finite and bounded on stress configs (heavy
        # outlier columns) where prior QuIP-style methods diverge / NaN.
        torch.manual_seed(1)
        w = torch.randn(64, 512)
        w[:, :5] *= 50.0
        recon = kashin_dct_quantize(w).dequantize(torch.float32)
        self.assertFalse(torch.isnan(recon).any())
        self.assertLess(_rel_err(recon, w), 0.4)

    def test_linear_method_wiring(self):
        # Exercises create_weights -> process_weights_after_loading -> apply and
        # asserts the served output tracks the full-precision linear.
        torch.manual_seed(0)
        method = KashinDCTLinearMethod(KashinDCTConfig())
        layer = torch.nn.Module()
        method.create_weights(layer, 512, [64], 512, 64, torch.float32)
        true_weight = torch.randn(64, 512) * 0.1
        layer.weight.data.copy_(true_weight)

        x = torch.randn(8, 512)
        reference = torch.nn.functional.linear(x, true_weight)
        method.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "kashin_dct_weight"))

        out = method.apply(layer, x)
        self.assertEqual(out.shape, reference.shape)
        self.assertLess(_rel_err(out, reference), 0.2)


if __name__ == "__main__":
    unittest.main()
