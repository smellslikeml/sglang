"""Token-adaptive expert skipping for MoE routing.

Adapted from *ACE: Adaptive Calibration-Free Expert Skipping for MoE-based
LLMs* (arXiv:2609.05228). Fixed top-k routing activates the same number of
expert slots for every token, so tokens that are well served by their top
expert still pay for the remaining routed slots. ACE's core insight is to
drop a routed slot only when *multiple complementary views* agree that the
expert contributes little to the token, while *always retaining the top-1
expert* -- this is what makes the skip robust where a single router-confidence
threshold is not.

This is a target-native adaptation (Mode 3): ACE's offline statistics -- the
Global Spectral Proxy (weight-derived per-expert capacity table) and the
Router-Conditioned Refinement direction prototypes -- require a model-load-time
pass over the expert weight matrices that ``select_experts`` does not have in
hand. They are intentionally out of scope here. Instead the two agreeing views
are computed online from signals already produced by routing:

  * **Gate-share view** -- the post-softmax weight of a slot relative to the
    token's strongest routed gate. A slot with a tiny share carries little of
    the token's expert mass.
  * **Router-direction view** -- the raw router logit of the selected expert
    measured against the per-token mean logit over *all* experts (a
    parameter-free stand-in for RCR's centered-direction alignment). A slot
    that barely clears the average expert is weakly preferred by the router,
    even if softmax renormalization inflated its share.

The two views are genuinely different -- a crowded logit distribution can give
a slot a healthy softmax share while its direction margin is thin, and vice
versa -- so requiring both to agree (plus the unconditional top-1 keep) mirrors
ACE's AND-gated decision rather than collapsing to naive gate thresholding.

A skipped slot has its weight zeroed (contribution removed on every backend)
and, on the CUDA/EP fused-MoE path, its id set to the invalid marker so the
kernel can drop the work entirely. Weights are not renormalized: the paper is
checkpoint-preserving and simply removes the slot's term from the weighted sum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKConfig

# Marker the CUDA / EP fused-MoE kernels treat as "no expert in this slot"
# (the same value ``_mask_topk_ids_padded_region`` fills padded rows with).
INVALID_EXPERT_ID = -1

_EPS = 1e-9


def should_apply_expert_skip(
    topk_config: TopKConfig,
    *,
    expert_location_dispatch_info: Optional[object],
    packed: bool,
) -> bool:
    """Whether adaptive expert skipping can be applied to this routing output.

    Kept deliberately narrow: the skip reads router logits indexed by the
    selected expert ids, so it only holds when ids are logical == physical
    (no EPLB remap) and there are no fused shared-expert columns appended to
    the routed slots. The experimental fused topk+pack carrier is also skipped
    because its packed ids are produced independently of ``topk_ids``.
    """
    if not envs.SGLANG_ENABLE_MOE_ADAPTIVE_EXPERT_SKIP.get():
        return False
    if packed or expert_location_dispatch_info is not None:
        return False
    if topk_config.num_fused_shared_experts != 0:
        return False
    # Need at least two routed slots -- the top-1 is always kept.
    return topk_config.top_k >= 2


def apply_adaptive_expert_skip(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_routed_slots: int,
    threshold: float,
    invalidate_ids: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero low-contribution routed slots that both views agree on.

    ``topk_ids`` / ``topk_weights`` are ``[num_tokens, k]`` and ``router_logits``
    is ``[num_tokens, num_experts]``. Only the first ``num_routed_slots`` columns
    are considered routed experts. Returns fresh ``(topk_ids, topk_weights)``
    tensors; the inputs are left untouched.
    """
    if threshold <= 0.0 or num_routed_slots < 2:
        return topk_ids, topk_weights

    r = num_routed_slots
    routed_weights = topk_weights[:, :r]
    routed_ids = topk_ids[:, :r].long().clamp_min(0)

    # View A: softmax share relative to the token's strongest routed gate.
    top_gate = routed_weights.max(dim=-1, keepdim=True).values
    low_gate = routed_weights < threshold * top_gate.clamp_min(_EPS)

    # View B: router-logit margin above the per-token mean over all experts
    # (parameter-free RCR direction proxy), relative to the strongest margin.
    mean_logit = router_logits.mean(dim=-1, keepdim=True)
    selected_logit = router_logits.gather(1, routed_ids)
    margin = selected_logit - mean_logit
    top_margin = margin.max(dim=-1, keepdim=True).values
    low_margin = margin < threshold * top_margin.clamp_min(_EPS)

    # Always retain the per-token top-1 routed slot (highest gate).
    top1_slot = routed_weights.argmax(dim=-1, keepdim=True)
    cols = torch.arange(r, device=topk_weights.device).unsqueeze(0)
    is_top1 = cols == top1_slot

    skip = low_gate & low_margin & ~is_top1
    if not bool(skip.any()):
        return topk_ids, topk_weights

    skip_full = torch.zeros_like(topk_weights, dtype=torch.bool)
    skip_full[:, :r] = skip

    new_weights = torch.where(skip_full, torch.zeros_like(topk_weights), topk_weights)
    new_ids = topk_ids
    if invalidate_ids:
        invalid = torch.full_like(topk_ids, INVALID_EXPERT_ID)
        new_ids = torch.where(skip_full, invalid, topk_ids)
    return new_ids, new_weights


def maybe_apply_adaptive_expert_skip(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    topk_config: TopKConfig,
    expert_location_dispatch_info: Optional[object],
    packed: bool,
    invalidate_ids: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Guard + apply. Returns the inputs unchanged when skipping does not apply."""
    if not should_apply_expert_skip(
        topk_config,
        expert_location_dispatch_info=expert_location_dispatch_info,
        packed=packed,
    ):
        return topk_ids, topk_weights
    return apply_adaptive_expert_skip(
        topk_ids,
        topk_weights,
        router_logits,
        num_routed_slots=topk_config.top_k,
        threshold=envs.SGLANG_MOE_EXPERT_SKIP_THRESHOLD.get(),
        invalidate_ids=invalidate_ids,
    )
