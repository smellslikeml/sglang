"""Tests for token-adaptive expert skipping (ACE, arXiv:2609.05228).

The pure-function cases pin down the AND-gated skip decision (both views must
agree, top-1 always kept). The ``select_experts`` cases exercise the wiring
edit in ``sglang.srt.layers.moe.topk`` -- that the routed path invokes the skip
only when the env flag is on and the config is eligible.
"""

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.adaptive_expert_skip import (
    INVALID_EXPERT_ID,
    apply_adaptive_expert_skip,
    should_apply_expert_skip,
)

# Import the wired call site from a non-new module.
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


@pytest.fixture(autouse=True)
def _set_dummy_server_args():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


def _softmax_topk(router_logits, top_k):
    weights = torch.softmax(router_logits, dim=-1)
    top_w, top_ids = weights.topk(top_k, dim=-1)
    return top_w.contiguous(), top_ids.to(torch.int32)


def test_apply_skips_slot_both_views_reject_and_keeps_top1():
    # Token 0: expert 3 is far below the mean logit AND carries a tiny softmax
    # share -> both views reject it, so it is the only skipped slot. Token 1:
    # every selected expert clears both views -> nothing is skipped.
    router_logits = torch.tensor(
        [
            [10.0, 9.0, 8.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    top_w, top_ids = _softmax_topk(router_logits, top_k=4)

    new_ids, new_w = apply_adaptive_expert_skip(
        top_ids,
        top_w,
        router_logits,
        num_routed_slots=4,
        threshold=0.1,
        invalidate_ids=True,
    )

    # Row 0: exactly the dominated expert (id 3) is zeroed and invalidated.
    row0_ids, row0_w = top_ids[0].tolist(), top_w[0]
    skipped_col = row0_ids.index(3)
    assert new_w[0, skipped_col] == 0.0
    assert new_ids[0, skipped_col] == INVALID_EXPERT_ID
    # Top-1 (id 0) and the other strong experts are untouched.
    for col, eid in enumerate(row0_ids):
        if eid != 3:
            assert new_w[0, col] == row0_w[col]
            assert new_ids[0, col] == top_ids[0, col]
    # Row 1: no slot dropped.
    assert torch.equal(new_w[1], top_w[1])
    assert torch.equal(new_ids[1], top_ids[1])


def test_top1_never_skipped_even_when_isolated():
    # A single dominant expert with the rest near zero: the runners-up look weak
    # on both views, but the top-1 slot must survive regardless.
    router_logits = torch.tensor([[20.0, 0.2, 0.1, 0.0, 0.0, 0.0]])
    top_w, top_ids = _softmax_topk(router_logits, top_k=3)
    top1_id = int(top_ids[0, 0])

    _, new_w = apply_adaptive_expert_skip(
        top_ids,
        top_w,
        router_logits,
        num_routed_slots=3,
        threshold=0.5,
        invalidate_ids=False,
    )
    top1_col = top_ids[0].tolist().index(top1_id)
    assert new_w[0, top1_col] > 0.0


def test_zero_threshold_is_noop():
    router_logits = torch.tensor([[10.0, 9.0, 0.0, 0.0]])
    top_w, top_ids = _softmax_topk(router_logits, top_k=2)
    new_ids, new_w = apply_adaptive_expert_skip(
        top_ids,
        top_w,
        router_logits,
        num_routed_slots=2,
        threshold=0.0,
        invalidate_ids=True,
    )
    assert torch.equal(new_w, top_w)
    assert torch.equal(new_ids, top_ids)


def test_guard_requires_flag_and_eligible_config():
    cfg = TopKConfig(top_k=4)
    with envs.SGLANG_ENABLE_MOE_ADAPTIVE_EXPERT_SKIP.override(False):
        assert not should_apply_expert_skip(
            cfg, expert_location_dispatch_info=None, packed=False
        )
    with envs.SGLANG_ENABLE_MOE_ADAPTIVE_EXPERT_SKIP.override(True):
        assert should_apply_expert_skip(
            cfg, expert_location_dispatch_info=None, packed=False
        )
        # Fused shared-expert columns, packed carriers, and EPLB remap are all
        # out of scope -- the skip must decline rather than corrupt them.
        assert not should_apply_expert_skip(
            TopKConfig(top_k=4, num_fused_shared_experts=1),
            expert_location_dispatch_info=None,
            packed=False,
        )
        assert not should_apply_expert_skip(
            cfg, expert_location_dispatch_info=None, packed=True
        )
        assert not should_apply_expert_skip(
            cfg, expert_location_dispatch_info=object(), packed=False
        )


def _run_select(router_logits, top_k):
    hidden = torch.zeros((router_logits.shape[0], 8), dtype=torch.float32)
    return select_experts(
        hidden_states=hidden,
        router_logits=router_logits.clone(),
        topk_config=TopKConfig(top_k=top_k, renormalize=True, torch_native=True),
        layer_id=0,
    )


def test_select_experts_drops_dominated_expert_when_enabled():
    router_logits = torch.tensor([[10.0, 9.0, 8.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    baseline = _run_select(router_logits, top_k=4)
    assert torch.all(baseline.topk_weights > 0.0), "baseline keeps every slot"

    with envs.SGLANG_ENABLE_MOE_ADAPTIVE_EXPERT_SKIP.override(True):
        out = _run_select(router_logits, top_k=4)

    ids = out.topk_ids[0].tolist()
    dominated_col = ids.index(3)
    assert out.topk_weights[0, dominated_col] == 0.0
    # Every other selected slot (including top-1) keeps a positive weight.
    kept = [c for c in range(4) if c != dominated_col]
    assert torch.all(out.topk_weights[0, kept] > 0.0)
