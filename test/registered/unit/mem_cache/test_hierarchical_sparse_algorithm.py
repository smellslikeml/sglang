"""Unit tests for the Hierarchical Sparse Attention (HSA) page-selection algorithm.

Exercises the wiring added in
`sglang.srt.mem_cache.sparsity.factory` (the `"hierarchical_sparse"` registry
entry) end-to-end on CPU:

- `_create_sparse_algorithm` resolves the registry key to a
  `HierarchicalSparseAlgorithm`.
- The compressed-landmark representation + scaled query/landmark dot-product
  selects the query-aligned history page, and the base class always retains the
  recent page.

Adapted from "Random Long-Context Access for Mamba via Hardware-aligned
Hierarchical Sparse Attention" (https://arxiv.org/abs/2504.16795).
"""

import json
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.sparsity.algorithms.hierarchical_sparse_algorithm import (
    HierarchicalSparseAlgorithm,
)
from sglang.srt.mem_cache.sparsity.factory import (
    _create_sparse_algorithm,
    _parse_sparse_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DEVICE = torch.device("cpu")
PAGE_SIZE = 2
HEAD_DIM = 4
NUM_TOKENS = 16


def _make_config(**overrides):
    extra = {
        "algorithm": "hierarchical_sparse",
        "page_size": PAGE_SIZE,
        "num_recent_pages": 1,
        "sparsity_ratio": 0.5,
    }
    extra.update(overrides)
    server_args = SimpleNamespace(hisparse_config=json.dumps(extra))
    return _parse_sparse_config(server_args)


def _one_hot_key_buffer():
    """Keys where every token in page p points along basis vector p."""
    buf = torch.zeros((NUM_TOKENS, 1, HEAD_DIM), dtype=torch.float32, device=DEVICE)
    for page in range(HEAD_DIM):
        for tok in (page * PAGE_SIZE, page * PAGE_SIZE + 1):
            buf[tok, 0, page] = 1.0
    return buf


def _wire_algorithm(algo, seq_len):
    key_buffer = _one_hot_key_buffer()
    token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda _layer: key_buffer)
    # Identity logical->physical mapping so page indices line up with storage.
    req_to_token = torch.arange(NUM_TOKENS, dtype=torch.long, device=DEVICE).view(
        1, NUM_TOKENS
    )
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    states = SimpleNamespace(
        repr_constructed=torch.zeros(1, dtype=torch.bool, device=DEVICE),
        last_constructed_page=torch.zeros(1, dtype=torch.long, device=DEVICE),
        prompt_lens=torch.tensor([seq_len], dtype=torch.long, device=DEVICE),
    )
    algo.initialize_representation_pool(
        start_layer=0,
        end_layer=1,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
        states=states,
    )
    return key_buffer


class TestHierarchicalSparseAlgorithm(CustomTestCase):
    def test_factory_registry_wires_hsa(self):
        """The `"hierarchical_sparse"` key resolves through the shared factory."""
        config = _make_config()
        algo = _create_sparse_algorithm(config, DEVICE)
        self.assertIsInstance(algo, HierarchicalSparseAlgorithm)
        self.assertEqual(algo.page_size, PAGE_SIZE)
        self.assertEqual(algo.num_recent_pages, 1)

    def test_selects_query_aligned_and_recent_pages(self):
        config = _make_config()
        algo = _create_sparse_algorithm(config, DEVICE)
        seq_len = 8  # 4 pages of size 2
        key_buffer = _wire_algorithm(algo, seq_len)

        req_pool_indices = torch.tensor([0], dtype=torch.long, device=DEVICE)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=DEVICE)
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: True,
                is_decode_or_idle=lambda: False,
            ),
            seq_lens=seq_lens,
        )

        # Prefill: build the compressed page landmarks.
        algo.construct_representations(
            0, req_pool_indices, seq_lens, key_buffer, forward_batch
        )
        self.assertTrue(bool(algo.states.repr_constructed[0]))
        # Page 2's landmark should be the basis-2 direction (masked mean of keys).
        torch.testing.assert_close(
            algo.page_k_mean[0][2, 0],
            torch.tensor([0.0, 0.0, 1.0, 0.0], device=DEVICE),
        )

        # A query pointing along basis-2 must favour history page 2.
        query = torch.zeros((1, 1, HEAD_DIM), dtype=torch.float32, device=DEVICE)
        query[0, 0, 2] = 10.0
        sparse_mask = torch.tensor([True], device=DEVICE)

        selected, lengths = algo.retrieve_topk(
            query, 0, req_pool_indices, sparse_mask, forward_batch=forward_batch
        )

        self.assertEqual(int(lengths[0]), 2)
        chosen = {int(x) for x in selected[0].tolist() if x >= 0}
        # Top history page (2) plus the always-retained recent page (3).
        self.assertEqual(chosen, {2, 3})

    def test_masked_requests_select_nothing(self):
        config = _make_config()
        algo = _create_sparse_algorithm(config, DEVICE)
        seq_len = 8
        key_buffer = _wire_algorithm(algo, seq_len)

        req_pool_indices = torch.tensor([0], dtype=torch.long, device=DEVICE)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=DEVICE)
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: True,
                is_decode_or_idle=lambda: False,
            ),
            seq_lens=seq_lens,
        )
        algo.construct_representations(
            0, req_pool_indices, seq_lens, key_buffer, forward_batch
        )

        query = torch.ones((1, 1, HEAD_DIM), dtype=torch.float32, device=DEVICE)
        sparse_mask = torch.tensor([False], device=DEVICE)
        selected, lengths = algo.retrieve_topk(
            query, 0, req_pool_indices, sparse_mask, forward_batch=forward_batch
        )
        self.assertEqual(int(lengths[0]), 0)


if __name__ == "__main__":
    unittest.main()
