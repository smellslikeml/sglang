# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""BWAP Phase 2a — gather-GEMM prune path (functional-equivalent, faster).

Phase 1 masks the activation (``Z * mask``) after computing all ``D_ff``
neurons, so it saves no compute. Phase 2a computes ONLY the ``k`` retained
neurons by gathering the retained slices of the FFN weights, then running
standard GEMMs on the smaller matrices — numerically identical to the Phase-1
masked path (fp reorder only), but touching ``(1 - sparsity)`` of the FFN work.

The gather is an ``index_select`` done once per mask refresh (amortized over the
``T_p`` sparse steps of a cycle); the GEMMs are plain ``F.linear`` on the
gathered weights. No custom kernel (the A1 variant).

SGLang gated-MLP layout (e.g. ``Qwen2MLP``):
    gate_up_proj  MergedColumnParallelLinear, weight [2*D_ff, hidden]
                  rows [0:D_ff] = gate, [D_ff:2*D_ff] = up
    act_fn        SiluAndMul  ->  silu(gate) * up  = Z
    down_proj     RowParallelLinear, weight [hidden, D_ff]

Eligibility is intentionally narrow (see ``fast_path_eligible``): unquantized
float weights, no bias, TP=1. Anything else falls back to the Phase-1 masked
path, which is always correct. Broadening (quantized gather, TP row-reduce) is
a follow-up.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def gather_ffn_weights(
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    keep_idx: torch.Tensor,
    d_ff: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize the retained weight slices (run once per mask refresh).

    ``gate_up_weight`` [2*D_ff, hidden], ``down_weight`` [hidden, D_ff],
    ``keep_idx`` [k] (indices into D_ff). Returns
    (``gate_up_k`` [2k, hidden], ``down_k`` [hidden, k]) — the gate and the
    matching up rows for each kept neuron, and the kept down columns.
    """
    merged = torch.cat([keep_idx, keep_idx + d_ff])
    gate_up_k = gate_up_weight.index_select(0, merged).contiguous()
    down_k = down_weight.index_select(1, keep_idx).contiguous()
    return gate_up_k, down_k


def fused_pruned_mlp(
    x: torch.Tensor, gate_up_k: torch.Tensor, down_k: torch.Tensor
) -> torch.Tensor:
    """Prune-step FFN forward over only the k retained neurons.

    Equivalent to ``down_proj(SiluAndMul(gate_up_proj(x)) * mask)`` for the
    ``mask`` whose kept indices produced ``gate_up_k`` / ``down_k``. Assumes
    bias-free linears and TP=1 (see ``fast_path_eligible``).
    """
    gate_up = F.linear(x, gate_up_k)  # [T, 2k]
    z_k = silu_and_mul(gate_up)  # [T, k]
    return F.linear(z_k, down_k)  # [T, hidden]


def fused_pruned_mlp_into(
    x: torch.Tensor,
    gate_up_k: torch.Tensor,
    down_k: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Same math as ``fused_pruned_mlp`` but writes the final projection into the
    caller-owned ``out`` (a persistent buffer allocated outside the CUDA-graph pool).

    This is the capture-safe variant. The dense path's ``down_proj`` is a
    ``RowParallelLinear`` whose output allocation is lifetime-managed by SGLang's
    graph machinery; a raw ``F.linear`` output instead lands in the shared graph
    pool and, because the FFN result outlives its layer (it feeds the next layer's
    residual add), gets reused/clobbered across the model's layers under a captured
    replay — producing garbage independent of the weight values. Writing the
    cross-layer-lived result into ``out`` keeps it stable across replays, matching
    how the captured dense path behaves. The gate/up and activation tensors stay
    pool-allocated: they are consumed within the op sequence, so they are never
    live across another layer's allocation.
    """
    gate_up = F.linear(x, gate_up_k)  # [T, 2k]  (pool; consumed immediately)
    z_k = silu_and_mul(gate_up)  # [T, k]     (pool; consumed immediately)
    torch.matmul(z_k, down_k.t(), out=out)  # [T, hidden] -> persistent buffer
    return out


def fused_pruned_mlp_pool_free(
    x: torch.Tensor,
    gate_up_k: torch.Tensor,
    down_k: torch.Tensor,
    gate_up_buf: torch.Tensor,
    z_buf: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Fully pool-free variant: every width-sensitive intermediate (gate_up, the
    activation z) and the output are written into caller-owned persistent buffers
    via in-place ops — NO tensor is allocated in the CUDA-graph pool.

    Reduced-width (k < D_ff) intermediates are sizes that appear nowhere else in
    the model, so the shared graph pool reuses/clobbers them across layers on
    replay (full-width == dense sizes, which the pool handles; that is the only
    case that worked). Writing them into persistent buffers keeps them stable
    across replay. silu is expanded to g*sigmoid(g) so it too runs in place.
    """
    torch.matmul(x, gate_up_k.t(), out=gate_up_buf)  # [T, 2k]
    d = gate_up_buf.shape[-1] // 2
    g = gate_up_buf[:, :d]  # [T, k]  gate
    up = gate_up_buf[:, d:]  # [T, k]  up
    torch.sigmoid(g, out=z_buf)  # z = sigmoid(g)
    z_buf.mul_(g).mul_(up)  # z = silu(g) * up, in place
    torch.matmul(z_buf, down_k.t(), out=out)  # [T, hidden] -> persistent buffer
    return out


def fast_path_eligible(gate_up_proj, down_proj, tp_size: int) -> bool:
    """Whether the gather-GEMM fast path is valid for this layer.

    Narrow by design — bias-free, unquantized float weights, single TP rank.
    A RowParallelLinear at TP>1 needs an all-reduce that raw ``F.linear``
    skips, and quantized weights cannot be cheaply row-gathered; both fall back
    to the Phase-1 masked path.
    """
    if tp_size != 1:
        return False
    for proj in (gate_up_proj, down_proj):
        w = proj.weight
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or not w.is_floating_point():
            return False
        if (
            proj.bias is not None
        ):  # SGLang linears always carry a .bias attr (None when disabled)
            return False
    return True
