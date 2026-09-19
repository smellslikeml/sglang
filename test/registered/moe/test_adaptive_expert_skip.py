"""Tests for token-adaptive expert skipping (ACE, arXiv:2609.05228).

Exercises both the standalone skip rule and its wiring into the existing
``select_experts`` routing entry point. The rule is pure torch, so these run on
CPU.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.adaptive_expert_skip import (
    apply_adaptive_expert_skip,
    maybe_apply_adaptive_expert_skip,
)
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")
register_cpu_ci(est_time=10, suite="base-b-test-cpu-arm64")


class TestAdaptiveExpertSkipRule(CustomTestCase):
    def test_agreement_gate_drops_only_when_both_views_agree(self):
        # Row: top-1 gate 0.80; slots at 0.03 and 0.02.
        ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        weights = torch.tensor([[0.80, 0.15, 0.03, 0.02]])

        # floor=0.05 flags slots {2,3}; rel=0.03125 -> threshold 0.025 flags {3}
        # only. The AND keeps slot 2 and drops slot 3 -- proves the two-view
        # agreement, not a single threshold, decides the skip.
        out_ids, out_w = apply_adaptive_expert_skip(
            ids, weights, gate_floor=0.05, rel_ratio=0.03125
        )
        self.assertEqual(out_ids[0].tolist(), [0, 1, 2, -1])
        self.assertEqual(out_w[0, 3].item(), 0.0)
        # Retained slots pass through with their exact original gate.
        self.assertTrue(torch.equal(out_w[0, :3], weights[0, :3]))

    def test_top1_is_never_dropped(self):
        # Every gate is below the floor; the top-1 slot must still survive.
        ids = torch.tensor([[5, 6]], dtype=torch.int32)
        weights = torch.tensor([[0.001, 0.0005]])
        out_ids, out_w = apply_adaptive_expert_skip(
            ids, weights, gate_floor=0.05, rel_ratio=0.0
        )
        self.assertEqual(out_ids[0].tolist(), [5, -1])
        self.assertEqual(out_w[0, 0].item(), weights[0, 0].item())
        self.assertEqual(out_w[0, 1].item(), 0.0)

    def test_no_enabled_view_is_noop(self):
        # Non-positive thresholds disable every view -> nothing is dropped.
        ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        weights = torch.tensor([[0.80, 0.15, 0.03, 0.02]])
        out_ids, out_w = apply_adaptive_expert_skip(
            ids, weights, gate_floor=0.0, rel_ratio=0.0
        )
        self.assertEqual(out_ids.tolist(), ids.tolist())
        self.assertEqual(out_w.tolist(), weights.tolist())

    def test_maybe_is_disabled_by_default(self):
        ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        weights = torch.tensor([[0.80, 0.15, 0.03, 0.02]])
        out_ids, _ = maybe_apply_adaptive_expert_skip(ids, weights)
        self.assertEqual(out_ids.tolist(), ids.tolist())

    def test_maybe_skips_fused_shared_expert_configs(self):
        # Shared-expert slots must always run, so the whole config is left alone.
        ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
        weights = torch.tensor([[0.80, 0.15, 0.03, 0.02]])
        with (
            envs.SGLANG_ENABLE_ADAPTIVE_EXPERT_SKIP.override(True),
            envs.SGLANG_ADAPTIVE_EXPERT_SKIP_GATE_FLOOR.override(0.05),
        ):
            out_ids, _ = maybe_apply_adaptive_expert_skip(
                ids, weights, num_fused_shared_experts=1
            )
        self.assertEqual(out_ids.tolist(), ids.tolist())


class TestSelectExpertsWiring(CustomTestCase):
    """The env-gated hook must be reachable through ``select_experts`` and leave
    the default (flag off) routing output untouched."""

    def _select(self, logits):
        def custom_routing(hidden_states, gating_output, topk, renormalize):
            w, ids = torch.topk(torch.softmax(gating_output, dim=-1), topk, dim=-1)
            return w.float(), ids.to(torch.int32)

        cfg = TopKConfig(
            top_k=4, renormalize=False, custom_routing_function=custom_routing
        )
        return select_experts(
            hidden_states=torch.randn(logits.shape[0], 16),
            router_logits=logits,
            topk_config=cfg,
            layer_id=0,
        )

    def _dominant_logits(self):
        # Softmax gates: ~0.83 / 0.11 / 7.5e-4 / 2.8e-4 for the top-4 experts.
        logits = torch.full((2, 8), 0.0)
        logits[:, 0] = 4.0  # top-1
        logits[:, 1] = 2.0  # meaningful contribution
        logits[:, 2] = -3.0  # negligible
        logits[:, 3] = -4.0  # negligible
        return logits

    def test_flag_off_leaves_routing_unchanged(self):
        logits = self._dominant_logits()
        out = self._select(logits)
        # No -1 sentinels, all four routed slots retained.
        self.assertTrue(torch.all(out.topk_ids >= 0))
        self.assertEqual(out.topk_ids.shape, (2, 4))

    def test_flag_on_drops_low_contribution_slots(self):
        logits = self._dominant_logits()
        with (
            envs.SGLANG_ENABLE_ADAPTIVE_EXPERT_SKIP.override(True),
            envs.SGLANG_ADAPTIVE_EXPERT_SKIP_GATE_FLOOR.override(0.02),
            envs.SGLANG_ADAPTIVE_EXPERT_SKIP_REL_RATIO.override(0.1),
        ):
            out = self._select(logits)
        # The two negligible experts are dropped for every token...
        self.assertTrue(torch.all(out.topk_ids[:, 2:] == -1))
        self.assertTrue(torch.all(out.topk_weights[:, 2:] == 0.0))
        # ...while the top-1 and the meaningful second expert are both kept.
        self.assertTrue(torch.all(out.topk_ids[:, :2] >= 0))
        self.assertTrue(torch.all(out.topk_weights[:, :2] > 0.0))


if __name__ == "__main__":
    unittest.main()
