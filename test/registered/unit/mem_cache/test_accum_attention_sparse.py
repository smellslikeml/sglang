"""Unit tests for the accumulated-attention sparse retrieval algorithm.

The tests go through the existing sparse-algorithm factory (registry wiring) and
exercise the ZSMerge-style behavior: a page that has accumulated attention over
past decode steps stays important even when the current query favors a different
page -- the signal Quest's current-query-only criticality cannot express.

    python -m pytest test/registered/unit/mem_cache/test_accum_attention_sparse.py -v
"""

from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.sparsity.algorithms.accum_attention_algorithm import (
    AccumAttentionAlgorithm,
)

# Import the wiring surface from NON-new modules to prove integration.
from sglang.srt.mem_cache.sparsity.core import SparseConfig
from sglang.srt.mem_cache.sparsity.factory import (
    _ALGORITHM_REGISTRY,
    _create_sparse_algorithm,
)


def _make_config(page_size=1, **extra):
    return SparseConfig(
        algorithm="accum_attention",
        page_size=page_size,
        sparse_extra_config=extra,
    )


def _make_pools(num_tokens, head_num, head_dim, max_req_tokens):
    """Minimal fake KV/req pools exposing only what the algorithm reads."""
    key_buf = torch.zeros(num_tokens, head_num, head_dim)
    token_pool = SimpleNamespace(get_key_buffer=lambda _layer: key_buf)
    # Identity mapping: logical token i -> physical token i.
    req_to_token = torch.arange(max_req_tokens, dtype=torch.long).unsqueeze(0)
    req_pool = SimpleNamespace(req_to_token=req_to_token)
    return token_pool, req_pool


class TestAccumAttentionRegistry(unittest.TestCase):
    def test_registered_in_factory(self):
        self.assertIn("accum_attention", _ALGORITHM_REGISTRY)

    def test_factory_resolves_to_algorithm(self):
        algo = _create_sparse_algorithm(_make_config(), torch.device("cpu"))
        self.assertIsInstance(algo, AccumAttentionAlgorithm)

    def test_rejects_bad_decay(self):
        with self.assertRaises(ValueError):
            _create_sparse_algorithm(_make_config(accum_decay=0.0), torch.device("cpu"))


class TestAccumulatedImportance(unittest.TestCase):
    """Core ZSMerge insight: history-weighted importance beats one-shot relevance."""

    def _fresh_algo(self, decay=0.9):
        algo = _create_sparse_algorithm(
            _make_config(page_size=1, accum_decay=decay), torch.device("cpu")
        )
        token_pool, req_pool = _make_pools(
            num_tokens=8, head_num=1, head_dim=4, max_req_tokens=8
        )
        algo.initialize_representation_pool(
            start_layer=0,
            end_layer=1,
            token_to_kv_pool=token_pool,
            req_to_token_pool=req_pool,
            states=None,
        )
        # Page 0 aligns with the "history" query, page 1 with the "current" query.
        algo.page_key_mean[0][0] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        algo.page_key_mean[0][1] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        algo.page_valid[0][0] = True
        algo.page_valid[0][1] = True
        return algo

    def _score(self, algo, query):
        phys_pages = torch.tensor([[0, 1]])
        scores = algo._retrieve_page_scores(
            layer_id=0,
            phys_pages=phys_pages,
            req_pool_indices=torch.tensor([0]),
            queries=query.view(1, 1, 4),
        )
        return scores[0, 0].item(), scores[0, 1].item()

    def test_history_outweighs_current_query(self):
        q_hist = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q_cur = torch.tensor([0.0, 1.0, 0.0, 0.0])

        # No history: a query favoring page 1 makes page 1 win.
        cold = self._fresh_algo()
        p0_cold, p1_cold = self._score(cold, q_cur)
        self.assertGreater(p1_cold, p0_cold)

        # With accumulated attention on page 0, the same current query cannot flip it.
        warm = self._fresh_algo()
        for _ in range(5):
            self._score(warm, q_hist)
        p0_warm, p1_warm = self._score(warm, q_cur)
        self.assertGreater(p0_warm, p1_warm)

    def test_invalid_pages_are_masked(self):
        algo = self._fresh_algo()
        algo.page_valid[0][1] = False  # invalidate page 1
        _, p1 = self._score(algo, torch.tensor([0.0, 1.0, 0.0, 0.0]))
        self.assertEqual(p1, float("-inf"))

    def test_decay_recency_emphasis(self):
        # Larger decay retains more history; page 0 keeps a higher lead.
        q_hist = torch.tensor([1.0, 0.0, 0.0, 0.0])
        q_cur = torch.tensor([0.0, 1.0, 0.0, 0.0])

        def lead_after_switch(decay):
            algo = self._fresh_algo(decay=decay)
            for _ in range(5):
                self._score(algo, q_hist)
            p0, p1 = self._score(algo, q_cur)
            return p0 - p1

        self.assertGreater(lead_after_switch(0.95), lead_after_switch(0.5))


class TestRetrieveTopkEndToEnd(unittest.TestCase):
    def test_recent_pages_always_retained(self):
        page_size = 2
        algo = _create_sparse_algorithm(
            _make_config(page_size=page_size, num_recent_pages=2, sparsity_ratio=0.5),
            torch.device("cpu"),
        )
        token_pool, req_pool = _make_pools(
            num_tokens=32, head_num=1, head_dim=4, max_req_tokens=32
        )
        algo.initialize_representation_pool(
            start_layer=0,
            end_layer=1,
            token_to_kv_pool=token_pool,
            req_to_token_pool=req_pool,
            states=None,
        )

        # Distinct per-page keys so scoring is well-defined.
        k_buffer = torch.randn(32, 1, 4)
        seq_len = 12
        num_pages = seq_len // page_size  # 6
        algo._compute_page_representations(
            layer_id=0,
            reqs=torch.tensor([0]),
            seq_lens=torch.tensor([seq_len]),
            start_page=0,
            end_page=torch.tensor([num_pages]),
            k_buffer=k_buffer,
        )

        forward_batch = SimpleNamespace(seq_lens=torch.tensor([seq_len]))
        out_indices, out_lengths = algo.retrieve_topk(
            queries=torch.randn(1, 1, 4),
            layer_id=0,
            req_pool_indices=torch.tensor([0]),
            sparse_mask=torch.tensor([True]),
            forward_batch=forward_batch,
        )

        self.assertEqual(out_indices.shape[0], 1)
        length = int(out_lengths[0].item())
        self.assertGreater(length, 0)

        selected = set(out_indices[0, :length].tolist())
        # Recent pages (4, 5) must always be retained.
        self.assertIn(num_pages - 1, selected)
        self.assertIn(num_pages - 2, selected)
        # No out-of-range or padding indices in the valid region.
        self.assertTrue(all(0 <= idx < num_pages for idx in selected))


if __name__ == "__main__":
    unittest.main()
