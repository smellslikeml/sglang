import unittest

from sglang.srt.speculative.adaptive_spec_params import AdaptiveStepSlot
from sglang.srt.speculative.dynamic_draft_length import (
    DraftConfidenceTracker,
    accept_profile,
    conditional_accept_profile,
    confidence_implied_length,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")


class TestConfidenceLengthRule(unittest.TestCase):
    def test_accept_profile_is_per_position_survival_fraction(self):
        # counts [3, 1, 2, 0]: pos0 accepted by 3/4 reqs (c>0), pos1 by 2/4
        # (c>1), pos2 by 1/4 (c>2), pos3 by none.
        self.assertEqual(
            accept_profile([3, 1, 2, 0], 4),
            [0.75, 0.5, 0.25, 0.0],
        )

    def test_empty_batch_profile_is_all_zero(self):
        self.assertEqual(accept_profile([], 3), [0.0, 0.0, 0.0])
        self.assertEqual(accept_profile([1, 2], 0), [])

    def test_length_stops_when_cumulative_confidence_drops(self):
        # cumulative product: 0.9, 0.72, 0.36. threshold 0.5 clears the first
        # two positions and truncates at the third.
        self.assertEqual(
            confidence_implied_length([0.9, 0.8, 0.5], threshold=0.5),
            2,
        )

    def test_high_confidence_keeps_full_length(self):
        self.assertEqual(
            confidence_implied_length([1.0, 1.0, 1.0], threshold=0.5),
            3,
        )

    def test_nonpositive_threshold_never_truncates(self):
        self.assertEqual(
            confidence_implied_length([0.1, 0.1], threshold=0.0),
            2,
        )

    def test_tracker_starts_optimistic_and_imposes_no_ceiling(self):
        tracker = DraftConfidenceTracker(num_positions=4, ema_alpha=0.2)
        self.assertEqual(tracker.implied_length(threshold=0.5), 4)

    def test_conditional_profile_telescopes_to_joint_survival(self):
        # A survival curve S=[1, 1, 1, 0.5, 0] yields conditionals whose running
        # product equals S at every prefix: the length-4 chain survives with
        # probability 0.5 (S[3]), not 1*1*1*0.5 vs a product-of-survivals.
        survival = [1.0, 1.0, 1.0, 0.5, 0.0]
        conditional = conditional_accept_profile(survival)
        self.assertEqual(conditional, [1.0, 1.0, 1.0, 0.5, 0.0])
        product = 1.0
        for cond, surv in zip(conditional, survival):
            product *= cond
            self.assertAlmostEqual(product, surv)

    def test_conditional_profile_guards_zero_survival(self):
        # Once survival hits zero the chain is dead; every later conditional is
        # zero rather than a divide-by-zero.
        self.assertEqual(
            conditional_accept_profile([0.5, 0.0, 0.0]),
            [0.5, 0.0, 0.0],
        )

    def test_flat_survival_keeps_full_length_not_product_of_survivals(self):
        # Counts [7, 0]: half the batch accepts every draft, giving a flat 0.5
        # survival at all positions. The joint survival of the whole chain is
        # therefore 0.5 (>= threshold), so the chain must NOT be truncated. The
        # naive product-of-survivals (0.5 ** 7) would over-truncate to length 1.
        tracker = DraftConfidenceTracker(num_positions=7, ema_alpha=1.0)
        tracker.update([7, 0])
        self.assertEqual(tracker.implied_length(threshold=0.5), 7)


class TestAdaptiveStepSlotConfidenceCeiling(unittest.TestCase):
    """DynaSD confidence ceiling wired into the post-verify step controller."""

    def _config(self, **overrides):
        cfg = {
            "candidate_steps": [1, 3, 7],
            "ema_alpha": 1.0,
            "warmup_batches": 0,
            "update_interval": 1,
            "up_hysteresis": 0.0,
            "down_hysteresis": 0.0,
        }
        cfg.update(overrides)
        return cfg

    def test_ceiling_caps_a_step_up_the_accept_ema_alone_would_allow(self):
        # counts [4, 4, 3, 3]: mean accept length 3.5 drives the accept-length
        # EMA to request the top step (7). But acceptance decays with depth —
        # joint survival is 1.0 through position 2 and 0.5 at position 3, dropping
        # to 0 after — so the confidence rule implies length 4, capping the step
        # selection at the largest candidate <= 4 (i.e. 3).
        without = AdaptiveStepSlot(initial_steps=1, cfg=self._config())
        self.assertTrue(without.update([4, 4, 3, 3]))
        self.assertEqual(without.current_steps, 7)

        with_conf = AdaptiveStepSlot(
            initial_steps=1, cfg=self._config(confidence_threshold=0.5)
        )
        self.assertTrue(with_conf.update([4, 4, 3, 3]))
        self.assertEqual(with_conf.current_steps, 3)

    def test_ceiling_does_not_restrict_when_confidence_is_high(self):
        with_conf = AdaptiveStepSlot(
            initial_steps=1, cfg=self._config(confidence_threshold=0.5)
        )
        # Every draft accepted: confidence stays 1.0 across all positions, so
        # the ceiling never binds and the accept EMA is free to scale up.
        self.assertTrue(with_conf.update([7, 7]))
        self.assertEqual(with_conf.current_steps, 7)

    def test_confidence_disabled_by_default(self):
        slot = AdaptiveStepSlot(initial_steps=3, cfg=self._config())
        self.assertIsNone(slot._confidence)
        self.assertEqual(slot.confidence_threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
