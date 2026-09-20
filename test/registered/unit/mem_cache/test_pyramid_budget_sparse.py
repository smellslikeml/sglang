"""Unit tests for the pyramid-budget sparse attention algorithm.

Exercises the wiring through the existing (non-new) sparsity factory registry
and the base-class layer-budget hook, plus the algorithm's own page scoring.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch

# Import from existing (non-new) modules (factory, core, Quest) to prove the
# integration wiring, alongside the new PyramidBudget algorithm.
from sglang.srt.mem_cache.sparsity.algorithms.pyramid_budget_algorithm import (
    PyramidBudgetAlgorithm,
)
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import SparseConfig
from sglang.srt.mem_cache.sparsity.factory import (
    _ALGORITHM_REGISTRY,
    _create_sparse_algorithm,
)


def _make_config(**extra):
    return SparseConfig(
        algorithm="pyramid_budget",
        page_size=1,
        sparse_extra_config=extra,
    )


class TestPyramidBudgetRegistration(unittest.TestCase):
    def test_registered_in_factory(self):
        self.assertIn("pyramid_budget", _ALGORITHM_REGISTRY)

    def test_factory_builds_pyramid_algorithm(self):
        algo = _create_sparse_algorithm(_make_config(), torch.device("cpu"))
        self.assertIsInstance(algo, PyramidBudgetAlgorithm)


class TestPyramidBudgetSchedule(unittest.TestCase):
    def _algo(self, **extra):
        algo = _create_sparse_algorithm(_make_config(**extra), torch.device("cpu"))
        algo.start_layer = 0
        algo.end_layer = 8
        return algo

    def test_budget_decreases_with_depth(self):
        algo = self._algo(pyramid_max_ratio=0.8, pyramid_min_ratio=0.2)
        ratios = [algo._sparsity_ratio_for_layer(l) for l in range(8)]
        # Monotonically non-increasing pyramid.
        for prev, nxt in zip(ratios, ratios[1:]):
            self.assertGreaterEqual(prev, nxt)
        # Endpoints hit the configured max/min.
        self.assertAlmostEqual(ratios[0], 0.8)
        self.assertAlmostEqual(ratios[-1], 0.2)

    def test_default_min_ratio_is_half_of_max(self):
        algo = self._algo(pyramid_max_ratio=0.6)
        self.assertAlmostEqual(algo.pyramid_min_ratio, 0.3)

    def test_invalid_ratios_rejected(self):
        with self.assertRaises(ValueError):
            self._algo(pyramid_max_ratio=0.3, pyramid_min_ratio=0.6)

    def test_base_default_ratio_is_uniform(self):
        # The base hook keeps Quest (and any non-overriding algorithm) uniform.
        quest = QuestAlgorithm(
            SparseConfig(page_size=1, sparse_extra_config={"sparsity_ratio": 0.4}),
            torch.device("cpu"),
        )
        self.assertEqual(quest._sparsity_ratio_for_layer(0), 0.4)
        self.assertEqual(quest._sparsity_ratio_for_layer(5), 0.4)


class TestPyramidBudgetScoring(unittest.TestCase):
    def test_query_aligned_page_scores_highest(self):
        algo = _create_sparse_algorithm(_make_config(), torch.device("cpu"))
        layer_id = 0
        # Three pages: page 1's mean key aligns with the query direction.
        algo.page_k_mean = {
            layer_id: torch.tensor(
                [
                    [[1.0, 0.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 1.0, 0.0]],
                ],
                dtype=torch.float32,
            )
        }
        algo.page_valid = {layer_id: torch.tensor([True, True, True])}

        phys_pages = torch.tensor([[0, 1, 2]])
        query = torch.tensor([[0.0, 5.0, 0.0, 0.0]])  # points at page 1
        scores = algo._retrieve_page_scores(
            layer_id, phys_pages, torch.tensor([0]), query
        )
        self.assertEqual(int(scores.argmax(dim=1).item()), 1)

    def test_invalid_pages_masked_to_neg_inf(self):
        algo = _create_sparse_algorithm(_make_config(), torch.device("cpu"))
        layer_id = 0
        algo.page_k_mean = {
            layer_id: torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=torch.float32)
        }
        algo.page_valid = {layer_id: torch.tensor([True, False])}
        scores = algo._retrieve_page_scores(
            layer_id,
            torch.tensor([[0, 1]]),
            torch.tensor([0]),
            torch.tensor([[1.0, 0.0]]),
        )
        self.assertTrue(torch.isfinite(scores[0, 0]))
        self.assertEqual(scores[0, 1].item(), float("-inf"))


if __name__ == "__main__":
    unittest.main()
