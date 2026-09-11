# SPDX-License-Identifier: Apache-2.0
"""Vector-quantized linear method with offline codebook expansion.

Adapted from "Unfolding the Leech Lattice: Fused Multi-Shell Decoding and VRAM
Layouts for 2-Bit LLM Weights" (arXiv:2609.02652). The paper's serving path for
lattice vector-quantized weights has three stages: weights are stored on disk as
codebook indices, expanded *offline* into a GPU layout at load time, and read by
a fused dequantize-plus-matvec GEMV during decode. This module ports that
structure at full fidelity while substituting two auxiliary components with
target-native equivalents:

  * the paper's hand-written fused multi-shell CUDA decode+matvec kernel is
    replaced by SGLang's native materialize-then-``F.linear`` path (the offline
    expansion produces a dense weight the existing GEMM consumes), and
  * the 301-class Leech-lattice codebook enumeration is replaced by an
    arbitrary codebook supplied in the checkpoint -- the decode path
    (index -> shell vector -> per-channel scale -> reshape) is codebook-agnostic.

What is kept faithful is the paper's first contribution -- the offline expansion
of a codebook-indexed weight into an in-VRAM layout consumed by the linear
apply -- and its second contribution: the in-VRAM rate is a design axis distinct
from the on-disk rate, which we surface as an observable bits-per-weight report
at expansion time. What is scoped out (and would be a downstream kernel PR) is
the fused warp-divergence-free multi-shell decode that lets the true ~2-bit rate
avoid a lookup table.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

# The Leech lattice lives in 24 dimensions; a group of this many contiguous
# weights along the input dim maps to a single codebook (shell) index.
DEFAULT_VEC_DIM = 24


def dequant_vector(
    qindices: torch.Tensor,
    codebook: torch.Tensor,
    weight_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    """Expand codebook indices into a dense weight (the offline decode).

    Args:
        qindices: ``[out, n_groups]`` integer indices into ``codebook``.
        codebook: ``[codebook_size, vec_dim]`` shell vectors.
        weight_scale: optional ``[out, 1]`` per-output-channel scale.

    Returns:
        Dense ``[out, n_groups * vec_dim]`` weight in ``codebook``'s dtype.
    """
    out, n_groups = qindices.shape
    vec_dim = codebook.shape[1]
    # Gather one vec_dim-wide shell vector per group, then unfold to a row.
    vectors = codebook[qindices.long()]
    weight = vectors.reshape(out, n_groups * vec_dim)
    if weight_scale is not None:
        weight = weight * weight_scale.to(weight.dtype)
    return weight.contiguous()


def bits_per_weight(codebook_size: int, vec_dim: int) -> float:
    """On-disk index rate: ``log2(codebook_size) / vec_dim`` bits per weight."""
    return math.log2(max(codebook_size, 1)) / vec_dim


class LeechVQConfig(QuantizationConfig):
    """Config for lattice vector-quantized (codebook-indexed) linear weights."""

    def __init__(
        self,
        vec_dim: int = DEFAULT_VEC_DIM,
        codebook_size: int = 256,
        has_weight_scale: bool = True,
        ignored_layers: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if vec_dim <= 0:
            raise ValueError(f"vec_dim must be positive, got {vec_dim}")
        if codebook_size <= 1:
            raise ValueError(f"codebook_size must be > 1, got {codebook_size}")
        self.vec_dim = vec_dim
        self.codebook_size = codebook_size
        self.has_weight_scale = has_weight_scale
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> str:
        return "leech_vq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # Pure-PyTorch decode + GEMM: no custom kernel capability floor.
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> LeechVQConfig:
        return cls(
            vec_dim=int(cls.get_from_keys_or(config, ["vec_dim"], DEFAULT_VEC_DIM)),
            codebook_size=int(cls.get_from_keys(config, ["codebook_size"])),
            has_weight_scale=bool(
                cls.get_from_keys_or(config, ["has_weight_scale"], True)
            ),
            ignored_layers=cls.get_from_keys_or(config, ["ignored_layers"], None),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return LeechVQLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class LeechVQLinearMethod(LinearMethodBase):
    """Serving path: codebook-indexed storage -> offline expand -> GEMV."""

    def __init__(self, quant_config: LeechVQConfig) -> None:
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
    ) -> None:
        vec_dim = self.quant_config.vec_dim
        if input_size_per_partition % vec_dim != 0:
            raise ValueError(
                f"input_size_per_partition ({input_size_per_partition}) must be "
                f"divisible by vec_dim ({vec_dim}) for lattice VQ."
            )
        output_size_per_partition = sum(output_partition_sizes)
        n_groups = input_size_per_partition // vec_dim

        # On-disk layout: one integer shell index per vec_dim-wide group.
        qindices = Parameter(
            torch.zeros(output_size_per_partition, n_groups, dtype=torch.int32),
            requires_grad=False,
        )
        set_weight_attrs(qindices, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("qindices", qindices)
        set_weight_attrs(qindices, extra_weight_attrs)

        # Codebook is shared across the partition (not sharded).
        codebook = Parameter(
            torch.zeros(self.quant_config.codebook_size, vec_dim, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("codebook", codebook)
        set_weight_attrs(codebook, extra_weight_attrs)

        if self.quant_config.has_weight_scale:
            weight_scale = Parameter(
                torch.ones(output_size_per_partition, 1, dtype=params_dtype),
                requires_grad=False,
            )
            set_weight_attrs(weight_scale, {"output_dim": 0})
            layer.register_parameter("weight_scale", weight_scale)
            set_weight_attrs(weight_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Offline expansion of the codebook into a dense in-VRAM layout."""
        has_scale = self.quant_config.has_weight_scale
        weight_scale = layer.weight_scale if has_scale else None
        weight = dequant_vector(layer.qindices, layer.codebook, weight_scale)
        layer.register_parameter("weight", Parameter(weight, requires_grad=False))

        # The in-VRAM rate is a design axis distinct from the on-disk rate:
        # the indices cost log2(K)/vec_dim bits/weight while the expanded dense
        # layout costs the element width. Surface both so the trade-off the
        # paper measures is observable at load time.
        disk_bpw = bits_per_weight(
            self.quant_config.codebook_size, self.quant_config.vec_dim
        )
        vram_bpw = weight.element_size() * 8
        logger.debug(
            "leech_vq expand: on-disk %.3f bits/weight -> in-VRAM %d bits/weight "
            "(%.2fx), weight shape %s",
            disk_bpw,
            vram_bpw,
            vram_bpw / max(disk_bpw, 1e-9),
            tuple(weight.shape),
        )

        # Free the compact source layout now that the dense one is live.
        source_params = ["qindices", "codebook"]
        if has_scale:
            source_params.append("weight_scale")
        for name in source_params:
            delattr(layer, name)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, bias)
