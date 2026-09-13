import unittest

from sglang.srt.speculative.adaptive_spec_params import AdaptiveStepSlot
from sglang.srt.speculative.dynamic_draft_length import (
    DraftConfidenceTracker,
    accept_profile,
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
        # counts [7, 0]: mean accept length 3.5 drives the accept-length EMA to
        # request the top step, but per-position confidence is a flat 0.5 whose
        # cumulative product falls below 0.5 after one position.
        without = AdaptiveStepSlot(initial_steps=3, cfg=self._config())
        self.assertTrue(without.update([7, 0]))
        self.assertEqual(without.current_steps, 7)

        with_conf = AdaptiveStepSlot(
            initial_steps=3, cfg=self._config(confidence_threshold=0.5)
        )
        self.assertTrue(with_conf.update([7, 0]))
        self.assertEqual(with_conf.current_steps, 1)

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
