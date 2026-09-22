"""Unit tests for RheoSampling proxy-score decoupling in the EAGLE draft path.

RheoSampling (arXiv:2609.21827) decouples the draft-tree pruning score from the
verification probability so stochastic (T>0) drafting keeps deep branches instead
of collapsing toward the greedy chain. These tests exercise the wiring through the
existing ``organize_draft_results`` entry point (not the new module alone), plus
the depth-normalized proxy transform itself.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.draft_tree_proxy_scores import (
    draft_node_depths,
    proxy_scores_enabled,
    temperature_proxy_scores,
)

# organize_draft_results is imported from the real, pre-existing draft call site
# to prove the integration wiring, not just the standalone helper.
from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _draft_tree_lists():
    """A 3-step, topk=2 draft tree (10 nodes) laid out by depth.

    Depth-1 nodes carry the highest raw scores and depth-3 the lowest, mirroring
    the product-of-probabilities shrink that RheoSampling counteracts. Token ids
    equal ``column_index * 10`` so the selected set is readable from the result.
    """
    score_list = [
        torch.tensor([[[0.9, 0.8]]]),  # depth 1 -> cols 0,1
        torch.tensor([[[0.5, 0.45], [0.4, 0.35]]]),  # depth 2 -> cols 2..5
        torch.tensor([[[0.30, 0.28], [0.26, 0.24]]]),  # depth 3 -> cols 6..9
    ]
    token_list = [
        torch.tensor([[0, 10]]),
        torch.tensor([[20, 30, 40, 50]]),
        torch.tensor([[60, 70, 80, 90]]),
    ]
    parents_list = [torch.zeros((1, 3), dtype=torch.long)]
    return score_list, token_list, parents_list


class TestDraftNodeDepths(unittest.TestCase):
    def test_depths_track_step_blocks(self):
        score_list, _, _ = _draft_tree_lists()
        depths = draft_node_depths(score_list)
        expected = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        self.assertTrue(torch.equal(depths, expected))


class TestTemperatureProxyScores(unittest.TestCase):
    def test_zero_temperature_is_identity(self):
        scores = torch.tensor([[0.9, 0.5, 0.3, 0.24]])
        depths = torch.tensor([1.0, 2.0, 3.0, 3.0])
        temps = torch.zeros((1, 1))
        proxy = temperature_proxy_scores(scores, temps, depths)
        self.assertTrue(torch.equal(proxy, scores))

    def test_depth_one_never_reweighted(self):
        scores = torch.tensor([[0.9, 0.8]])
        depths = torch.tensor([1.0, 1.0])
        temps = torch.tensor([[5.0]])
        proxy = temperature_proxy_scores(scores, temps, depths)
        self.assertTrue(torch.allclose(proxy, scores))

    def test_positive_temperature_boosts_deeper_nodes(self):
        # A deep node with a lower raw score can overtake a shallower node.
        scores = torch.tensor([[0.35, 0.30]])  # col0 depth 2, col1 depth 3
        depths = torch.tensor([2.0, 3.0])
        temps = torch.tensor([[3.0]])
        proxy = temperature_proxy_scores(scores, temps, depths)
        # Raw ordering favors col0; the depth-normalized proxy flips it.
        self.assertGreater(scores[0, 0].item(), scores[0, 1].item())
        self.assertGreater(proxy[0, 1].item(), proxy[0, 0].item())

    def test_mixed_batch_keeps_greedy_row_exact(self):
        scores = torch.tensor([[0.5, 0.25], [0.5, 0.25]])
        depths = torch.tensor([2.0, 3.0])
        temps = torch.tensor([[0.0], [2.0]])
        proxy = temperature_proxy_scores(scores, temps, depths)
        self.assertTrue(torch.equal(proxy[0], scores[0]))  # greedy row untouched
        self.assertFalse(torch.equal(proxy[1], scores[1]))  # stochastic row moved


class TestOrganizeDraftResultsProxy(unittest.TestCase):
    def test_greedy_temperatures_match_no_proxy(self):
        score_list, token_list, parents_list = _draft_tree_lists()
        baseline = organize_draft_results(score_list, token_list, parents_list, 7)

        score_list, token_list, parents_list = _draft_tree_lists()
        gated = organize_draft_results(
            score_list,
            token_list,
            parents_list,
            7,
            proxy_temperatures=torch.zeros((1, 1)),
        )
        # T=0 must reproduce the deterministic path exactly.
        self.assertTrue(torch.equal(baseline[1], gated[1]))
        self.assertTrue(torch.equal(baseline[2], gated[2]))

    def test_stochastic_temperature_selects_deeper_branch(self):
        score_list, token_list, parents_list = _draft_tree_lists()
        _, base_index, base_tokens = organize_draft_results(
            score_list, token_list, parents_list, 7
        )

        score_list, token_list, parents_list = _draft_tree_lists()
        _, proxy_index, proxy_tokens = organize_draft_results(
            score_list,
            token_list,
            parents_list,
            7,
            proxy_temperatures=torch.tensor([[3.0]]),
        )

        # Deterministic pruning keeps the shallow depth-2 node (token 50); the
        # temperature-gated proxy swaps in a deeper depth-3 node (token 60).
        self.assertIn(50, base_tokens.flatten().tolist())
        self.assertNotIn(60, base_tokens.flatten().tolist())
        self.assertIn(60, proxy_tokens.flatten().tolist())
        self.assertNotIn(50, proxy_tokens.flatten().tolist())
        self.assertFalse(torch.equal(base_index, proxy_index))


class TestProxyScoresGate(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertFalse(proxy_scores_enabled())

    def test_env_override_enables(self):
        with envs.SGLANG_ENABLE_RHEO_PROXY_SCORES.override(True):
            self.assertTrue(proxy_scores_enabled())
        self.assertFalse(proxy_scores_enabled())


if __name__ == "__main__":
    unittest.main()
