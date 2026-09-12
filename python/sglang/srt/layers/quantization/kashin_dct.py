# SPDX-License-Identifier: Apache-2.0
"""Kashin-DCT weight-only quantization.

Adapted (Mode 2 / adapted port) from "Structured Transforms for Low-Overhead
Quantization of Language Models" (arXiv:2609.11687). Each weight row is
factored into two bounded-``l-infinity`` components -- one in the identity
domain, one in the domain of a *sign-randomized DCT* -- and each factor is
clustered to a symmetric 2-bit codebook (two 2-bit factors == 4 bits/weight).

The sign-randomized DCT replaces the paper's dense random orthogonal matrix,
dropping the per-iteration transform cost from ``O(N^2)`` to ``O(N log N)`` while
keeping the bounded-``l-infinity`` (four-peak) factor distribution that makes
2-bit clustering stable. Cluster centers use the paper's closed-form symmetric
init, refined with a few Lloyd steps (the k-means multi-restart bottleneck is
removed). A greedy alternating update lets the second factor absorb the first
factor's quantization error, which is what keeps 2x2-bit competitive with a
naive 4-bit baseline.

Scoped out relative to the paper (auxiliary components, per Mode-2): OPTQ-style
sequential Hessian error compensation and QuIP incoherence preprocessing (both
need calibration data and belong in an offline checkpoint-prep pipeline), and a
native-2-bit GEMM kernel -- this integration decodes the factors back to the
compute dtype at load time and serves through the standard linear path.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import msgspec
import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

# Symmetric four-peak codebook offsets used for the closed-form center init.
# Centers per row are {-b, -a, a, b}; init a = m/3, b = m (m = row max-abs).
_CODEBOOK_INIT = (1.0 / 3.0, 1.0)


def orthonormal_dct(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal DCT-II along the last dim via FFT (``O(N log N)``)."""
    shape = x.shape
    n = shape[-1]
    x = x.contiguous().view(-1, n)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    k = -torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * n)
    out = vc[:, :, 0] * torch.cos(k) - vc[:, :, 1] * torch.sin(k)
    out[:, 0] /= math.sqrt(n) * 2
    out[:, 1:] /= math.sqrt(n / 2) * 2
    return (2 * out).view(shape)


def orthonormal_idct(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`orthonormal_dct` (orthonormal DCT-III)."""
    shape = x.shape
    n = shape[-1]
    xv = x.contiguous().view(-1, n) / 2
    xv[:, 0] *= math.sqrt(n) * 2
    xv[:, 1:] *= math.sqrt(n / 2) * 2
    k = torch.arange(n, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * n)
    v_i = torch.cat([xv[:, :1] * 0, -xv.flip([1])[:, :-1]], dim=1)
    real = xv * torch.cos(k) - v_i * torch.sin(k)
    imag = xv * torch.sin(k) + v_i * torch.cos(k)
    v = torch.fft.irfft(torch.complex(real, imag), n=n, dim=1)
    out = v.new_zeros(v.shape)
    out[:, ::2] += v[:, : n - (n // 2)]
    out[:, 1::2] += v.flip([1])[:, : n // 2]
    return out.view(shape)


def _make_signs(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic +/-1 vector seeding the sign-randomized DCT (reproducible)."""
    gen = torch.Generator().manual_seed(n)
    bits = torch.randint(0, 2, (n,), generator=gen)
    return torch.where(bits.bool(), 1.0, -1.0).to(device=device, dtype=dtype)


def _forward_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Q @ x  where Q is the sign-randomized DCT."""
    return orthonormal_dct(signs * x)


def _inverse_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Q^T @ x  (Q orthogonal, so this is the adjoint / inverse)."""
    return signs * orthonormal_idct(x)


def _two_bit_codebook(x: torch.Tensor, lloyd_iters: int) -> tuple:
    """Symmetric 4-level (2-bit) quantizer with closed-form + Lloyd centers.

    Returns ``(codes, centers)`` where ``codes`` in {0,1,2,3} index the per-row
    ``centers`` of shape ``(rows, 4)`` ordered ``[-b, -a, a, b]``.
    """
    ax = x.abs()
    m = ax.amax(dim=-1, keepdim=True).clamp_min(1e-12)
    a = m * _CODEBOOK_INIT[0]
    b = m * _CODEBOOK_INIT[1]
    for _ in range(lloyd_iters):
        to_a = (ax - a).abs() <= (ax - b).abs()
        na = to_a.sum(-1, keepdim=True).clamp_min(1)
        nb = (~to_a).sum(-1, keepdim=True).clamp_min(1)
        a = (ax * to_a).sum(-1, keepdim=True) / na
        b = (ax * ~to_a).sum(-1, keepdim=True) / nb
    centers = torch.cat([-b, -a, a, b], dim=-1)
    dist = (x.unsqueeze(-1) - centers.unsqueeze(-2)).abs()
    codes = dist.argmin(-1).to(torch.uint8)
    return codes, centers


class KashinDCTWeight(msgspec.Struct):
    """Compact two-factor 2-bit representation of one weight matrix."""

    codes_a: Any  # uint8 (rows, N), identity-domain factor
    codes_b: Any  # uint8 (rows, N), transform-domain factor
    centers_a: Any  # (rows, 4)
    centers_b: Any  # (rows, 4)
    signs: Any  # (N,) +/-1

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        a = torch.gather(self.centers_a, -1, self.codes_a.long())
        b = torch.gather(self.centers_b, -1, self.codes_b.long())
        w = a + _forward_transform(b, self.signs)
        return w.to(dtype)

    def bits_per_weight(self) -> float:
        n_weights = self.codes_a.numel()
        code_bits = 2 * 2 * n_weights  # two 2-bit factors
        center_bits = 32 * (self.centers_a.numel() + self.centers_b.numel())
        sign_bits = self.signs.numel()  # 1 bit each
        return (code_bits + center_bits + sign_bits) / n_weights


def kashin_dct_quantize(
    weight: torch.Tensor,
    *,
    num_iters: int = 15,
    kashin_const: float = 1.2,
    lloyd_iters: int = 3,
) -> KashinDCTWeight:
    """Factor + 2-bit quantize a ``(rows, N)`` weight along its last dim."""
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")
    w = weight.to(torch.float32)
    n = w.shape[-1]
    signs = _make_signs(n, w.device, w.dtype)

    # Kashin alternating projection: telescope W into A (identity domain) and
    # B (transform domain), each with bounded l-infinity norm.
    a_acc = torch.zeros_like(w)
    b_acc = torch.zeros_like(w)
    residual = w.clone()
    sqrt_n = math.sqrt(n)
    for _ in range(num_iters):
        level = kashin_const * residual.norm(dim=-1, keepdim=True) / sqrt_n
        clipped = torch.clamp(residual, -level, level)
        a_acc = a_acc + clipped
        residual = residual - clipped
        t = _inverse_transform(residual, signs)
        level = kashin_const * t.norm(dim=-1, keepdim=True) / sqrt_n
        clipped = torch.clamp(t, -level, level)
        b_acc = b_acc + clipped
        residual = residual - _forward_transform(clipped, signs)
    a_acc = a_acc + residual

    # Greedy alternating update: quantize A, then re-solve B against the
    # quantized-A residual so B absorbs A's 2-bit error before it is quantized.
    codes_a, centers_a = _two_bit_codebook(a_acc, lloyd_iters)
    a_q = torch.gather(centers_a, -1, codes_a.long())
    b_target = _inverse_transform(w - a_q, signs)
    codes_b, centers_b = _two_bit_codebook(b_target, lloyd_iters)
    return KashinDCTWeight(
        codes_a=codes_a,
        codes_b=codes_b,
        centers_a=centers_a,
        centers_b=centers_b,
        signs=signs,
    )


class KashinDCTLinearMethod(LinearMethodBase):
    """Linear method that Kashin-DCT quantizes weights at load time."""

    def __init__(self, quant_config: KashinDCTConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        params_dtype = layer.weight.dtype
        quantized = kashin_dct_quantize(
            layer.weight.data,
            num_iters=self.quant_config.num_iters,
            kashin_const=self.quant_config.kashin_const,
            lloyd_iters=self.quant_config.lloyd_iters,
        )
        layer.kashin_dct_weight = quantized
        logger.debug(
            "Kashin-DCT quantized a %s weight to %.2f bits/weight",
            tuple(layer.weight.shape),
            quantized.bits_per_weight(),
        )
        # Native-2-bit GEMM is scoped out: decode to the compute dtype and serve
        # through the standard dense linear path.
        layer.weight = Parameter(
            quantized.dequantize(params_dtype), requires_grad=False
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, bias)


class KashinDCTConfig(QuantizationConfig):
    """Config for online Kashin-DCT 2-bit-factor weight quantization."""

    def __init__(
        self,
        num_iters: int = 15,
        kashin_const: float = 1.2,
        lloyd_iters: int = 3,
    ) -> None:
        super().__init__()
        self.num_iters = num_iters
        self.kashin_const = kashin_const
        self.lloyd_iters = lloyd_iters

    @classmethod
    def get_name(cls) -> str:
        return "kashin_dct"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> KashinDCTConfig:
        return cls(
            num_iters=int(config.get("num_iters", 15)),
            kashin_const=float(config.get("kashin_const", 1.2)),
            lloyd_iters=int(config.get("lloyd_iters", 3)),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            return KashinDCTLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []
