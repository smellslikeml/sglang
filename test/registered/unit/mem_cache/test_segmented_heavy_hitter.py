"""Unit tests for the segmented heavy-hitter sparse attention algorithm.

Covers both the factory wiring and the retrieval behavior:
- The ``segmented_heavy_hitter`` key is registered in the sparsity factory and
  ``_create_sparse_algorithm`` dispatches it to ``SegmentedHeavyHitterAlgorithm``.
- ``retrieve_topk`` always keeps the leading attention-sink pages and the local
  window, produces a strictly sparse selection, and surfaces a mid-context
  heavy hitter (segment-local top scorer) — the BUZZ retention structure.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.sparsity.algorithms.segmented_heavy_hitter import (
    SegmentedHeavyHitterAlgorithm,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    RequestTrackers,
    SparseConfig,
)
from sglang.srt.mem_cache.sparsity.factory import (
    _ALGORITHM_REGISTRY,
    _create_sparse_algorithm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_TOTAL_TOKENS = 32
_HEAD_NUM = 1
_HEAD_DIM = 4
_SEQ_LEN = 20
_HEAVY_PAGE = 10  # mid-context page (page_size=1) made a strong heavy hitter


def _build_config():
    return SparseConfig(
        page_size=1,
        algorithm="segmented_heavy_hitter",
        sparse_extra_config={
            "segment_pages": 4,
            "num_sink_pages": 2,
            "num_recent_pages": 2,
            "sparsity_ratio": 0.5,
            "beehive_decay": 1.0,
        },
    )


def _build_wired_algorithm(config, device):
    """Create the algorithm via the factory and wire it to minimal fake pools."""
    algo = _create_sparse_algorithm(config, device)

    key_buf = torch.zeros(_TOTAL_TOKENS, _HEAD_NUM, _HEAD_DIM, device=device)
    key_buf[_HEAVY_PAGE] = 5.0  # aligns with the all-ones probe query -> top score

    token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda layer_id: key_buf)
    # Identity logical->physical token mapping for a single request.
    req_to_token = torch.arange(_TOTAL_TOKENS, device=device).unsqueeze(0)
    req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

    states = RequestTrackers(
        max_pool_size=1,
        device=device,
        num_layers=1,
        min_sparse_prompt_len=0,
        max_context_len=_TOTAL_TOKENS,
    )
    states.register(idx=0, prompt_len=_SEQ_LEN)

    algo.initialize_representation_pool(
        start_layer=0,
        end_layer=1,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
        states=states,
    )
    return algo, key_buf


class TestSegmentedHeavyHitter(CustomTestCase):
    def test_factory_registers_algorithm(self):
        self.assertIn("segmented_heavy_hitter", _ALGORITHM_REGISTRY)
        algo = _create_sparse_algorithm(_build_config(), torch.device("cpu"))
        self.assertIsInstance(algo, SegmentedHeavyHitterAlgorithm)

    def test_retrieval_keeps_sinks_window_and_heavy_hitter(self):
        device = torch.device("cpu")
        config = _build_config()
        algo, key_buf = _build_wired_algorithm(config, device)

        req_pool_indices = torch.tensor([0], device=device)
        seq_lens = torch.tensor([_SEQ_LEN], device=device)
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            seq_lens=seq_lens,
        )

        # Prefill: construct page representations for the whole prompt.
        algo.construct_representations(
            layer_id=0,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            k_buffer=key_buf,
            forward_batch=forward_batch,
        )

        queries = torch.ones(1, _HEAD_NUM * _HEAD_DIM, device=device)
        out_indices, out_lengths = algo.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool_indices,
            sparse_mask=torch.tensor([True], device=device),
            forward_batch=forward_batch,
        )

        length = int(out_lengths[0].item())
        self.assertGreater(length, 0)
        # Selection must be strictly sparse relative to the full page set.
        num_pages = _SEQ_LEN  # page_size == 1
        self.assertLess(length, num_pages)

        selected = set(out_indices[0, :length].tolist())
        # Attention-sink pages (first two) are always retained.
        self.assertTrue({0, 1}.issubset(selected))
        # Local window (last two pages) is always retained.
        self.assertTrue({num_pages - 2, num_pages - 1}.issubset(selected))
        # The mid-context heavy hitter is recovered by its segment.
        self.assertIn(_HEAVY_PAGE, selected)
        # Padding beyond the valid length stays at -1.
        self.assertTrue(bool((out_indices[0, length:] == -1).all().item()))

    def test_no_sparsification_when_short(self):
        """When the prompt fits within sink + window, no pages are dropped
        (empty selection == full attention)."""
        device = torch.device("cpu")
        config = _build_config()
        algo, key_buf = _build_wired_algorithm(config, device)

        short_len = 3  # <= num_sink_pages (2) + num_recent_pages (2)
        seq_lens = torch.tensor([short_len], device=device)
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            seq_lens=seq_lens,
        )
        queries = torch.ones(1, _HEAD_NUM * _HEAD_DIM, device=device)
        _, out_lengths = algo.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=torch.tensor([0], device=device),
            sparse_mask=torch.tensor([True], device=device),
            forward_batch=forward_batch,
        )
        self.assertEqual(int(out_lengths[0].item()), 0)


if __name__ == "__main__":
    unittest.main()
