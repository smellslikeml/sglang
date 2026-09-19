# Copyright 2024 SGLang Team
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
"""Token-adaptive expert skipping for MoE routing.

Adapted from *ACE: Adaptive Calibration-Free Expert Skipping for MoE-based
LLMs* (arXiv:2609.05228). ACE's insight is that fixed top-k routing activates
the same number of expert slots for every token even though many routed experts
contribute little, so a training-free / calibration-free rule can drop the
low-contribution slots at inference and reclaim the redundant compute.

Two structural invariants from the paper are reproduced here at full fidelity:

* **Agreement gating.** A slot is dropped only when *every* enabled view judges
  it low-contribution (the paper's "skip only when both views identify it as
  low-contribution"). This conservative AND is what separates ACE from a single
  router-confidence threshold, which the paper shows is unreliable.
* **Top-1 preservation.** The highest-gate expert in each row is never dropped,
  matching "while always retaining the top-1 expert".

Scoped substitution (this is an inspired, target-native experiment, not a
direct port): ACE's two offline signals -- the Global Spectral Proxy (per-expert
transformation capacity from the coupled gate/up/down projections) and the
Router-Conditioned Refinement prototypes -- are computed from model weights by a
separate offline profiling pass that SGLang does not host. In their place this
module uses two *runtime-available* views of the router gate that need no
calibration data and no extra tables:

* an absolute gate floor, and
* a gate relative to the per-token top-1 gate.

Both views operate on ``topk_weights`` already produced by ``select_experts``,
so the online cost stays "lightweight scalar operations" as the paper requires.
The offline GSP/RCR tables are intentionally out of scope; wiring them in would
require the per-model profiling infrastructure the repo lacks.

The feature is opt-in via env vars (see ``sglang.srt.environ``) and defaults to
a no-op.
"""

from __future__ import annotations

from typing import Tuple

import torch

from sglang.srt.environ import envs

# Sentinel written into ``topk_ids`` to mark a dropped expert slot. This mirrors
# the padded-region masking convention already used in ``topk.py`` (``-1`` on
# every downstream EP dispatch path means "no expert"), so no new contract is
# introduced for the fused MoE kernels to honor.
_SKIPPED_EXPERT_ID = -1


def adaptive_expert_skip_enabled() -> bool:
    """Whether token-adaptive expert skipping is turned on for this run."""
    return envs.SGLANG_ENABLE_ADAPTIVE_EXPERT_SKIP.get()


def _low_contribution_mask(
    topk_weights: torch.Tensor,
    *,
    gate_floor: float,
    rel_ratio: float,
) -> torch.Tensor:
    """Return a boolean mask (same shape as ``topk_weights``) that is ``True``
    for slots every enabled view agrees are low-contribution.

    A view with a non-positive threshold is disabled and casts no vote; when no
    view is enabled the mask is all-``False`` (nothing is dropped).
    """
    votes = []
    if gate_floor > 0.0:
        votes.append(topk_weights < gate_floor)
    if rel_ratio > 0.0:
        row_top1 = topk_weights.amax(dim=-1, keepdim=True)
        votes.append(topk_weights < rel_ratio * row_top1)

    if not votes:
        return torch.zeros_like(topk_weights, dtype=torch.bool)

    # Agreement gate: a slot is low-contribution only if EVERY enabled view says
    # so. This is ACE's conservative two-view AND, not a single threshold.
    agreed = votes[0]
    for vote in votes[1:]:
        agreed = agreed & vote
    return agreed


def apply_adaptive_expert_skip(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    gate_floor: float,
    rel_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop low-contribution, non-top-1 expert slots per token.

    Dropped slots get ``topk_ids = -1`` and ``topk_weights = 0`` so they neither
    dispatch nor contribute to the weighted expert sum. The per-token top-1
    expert is always retained. Inputs are not mutated in place.
    """
    if topk_ids.numel() == 0 or topk_ids.shape[-1] <= 1:
        # Nothing to skip: no tokens, or only the (protected) top-1 slot exists.
        return topk_ids, topk_weights

    low = _low_contribution_mask(
        topk_weights, gate_floor=gate_floor, rel_ratio=rel_ratio
    )
    if not bool(low.any()):
        return topk_ids, topk_weights

    # Protect the top-1 expert of every row from being dropped.
    top1_slot = topk_weights.argmax(dim=-1, keepdim=True)
    is_top1 = torch.zeros_like(low)
    is_top1.scatter_(-1, top1_slot, True)
    drop = low & ~is_top1

    new_ids = torch.where(drop, topk_ids.new_full((), _SKIPPED_EXPERT_ID), topk_ids)
    new_weights = torch.where(drop, topk_weights.new_zeros(()), topk_weights)
    return new_ids, new_weights


def maybe_apply_adaptive_expert_skip(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_fused_shared_experts: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Entry point wired into ``select_experts``.

    A no-op unless ``SGLANG_ENABLE_ADAPTIVE_EXPERT_SKIP`` is set. Configs with
    fused shared experts are left untouched: those appended slots must always
    run, and disentangling them from routed slots is out of scope for this
    experiment.
    """
    if not adaptive_expert_skip_enabled() or num_fused_shared_experts > 0:
        return topk_ids, topk_weights

    return apply_adaptive_expert_skip(
        topk_ids,
        topk_weights,
        gate_floor=envs.SGLANG_ADAPTIVE_EXPERT_SKIP_GATE_FLOOR.get(),
        rel_ratio=envs.SGLANG_ADAPTIVE_EXPERT_SKIP_REL_RATIO.get(),
    )
