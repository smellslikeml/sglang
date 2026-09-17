"""Unit tests for the ECHO-style bonus-logit corpus seeding used by the NGRAM
speculative worker (arXiv:2609.17241).

The pure selection/seed-building logic runs standalone; a wiring test imports
the existing NGRAM worker module to prove the integration is actually hooked up.
Both are skipped cleanly when the heavy runtime deps (numpy/torch/sgl_kernel)
are unavailable, so the file is collectible in a bare environment.
"""

import unittest

import pytest

# Importing anything under ``sglang`` pulls the package init (numpy/torch/...);
# skip the whole module rather than error when those are absent.
blc = pytest.importorskip(
    "sglang.srt.speculative.bonus_logit_candidates",
    reason="requires sglang runtime dependencies",
)


class TestBonusLogitCandidates(unittest.TestCase):
    def test_filter_keeps_only_confident_tokens(self):
        # Rows are sorted high-to-low; filtering stops at the first sub-floor.
        kept = blc.filter_confident_candidates(
            topk_tokens=[[10, 11, 12], [20, 21, 22]],
            topk_probs=[[0.6, 0.3, 0.02], [0.9, 0.05, 0.01]],
            prob_threshold=0.1,
        )
        self.assertEqual(kept, [[10, 11], [20]])

    def test_filter_dedups_within_request(self):
        kept = blc.filter_confident_candidates(
            topk_tokens=[[5, 5, 6]],
            topk_probs=[[0.5, 0.4, 0.2]],
            prob_threshold=0.1,
        )
        self.assertEqual(kept, [[5, 6]])

    def test_build_seeds_trims_prefix_to_trie_depth(self):
        seeds = blc.build_corpus_seed_sequences(
            context_tails=[[1, 2, 3, 4]],
            candidate_tokens=[[99]],
            max_trie_depth=3,
        )
        # prefix budget = max_trie_depth - 1 = 2 -> last two context tokens.
        self.assertEqual(seeds, [[3, 4, 99]])

    def test_empty_candidates_produce_no_seeds(self):
        seeds = blc.build_corpus_seed_sequences(
            context_tails=[[9, 8, 7]],
            candidate_tokens=[[]],
            max_trie_depth=4,
        )
        self.assertEqual(seeds, [])

    def test_end_to_end_seeds(self):
        seeds = blc.bonus_logit_corpus_seeds(
            topk_tokens=[[10, 11, 12], [20, 21, 22]],
            topk_probs=[[0.6, 0.3, 0.02], [0.9, 0.05, 0.01]],
            context_tails=[[1, 2, 3, 4], [7, 8]],
            prob_threshold=0.1,
            max_trie_depth=3,
        )
        self.assertEqual(seeds, [[3, 4, 10], [3, 4, 11], [7, 8, 20]])

    def test_mismatched_row_counts_raise(self):
        with self.assertRaises(ValueError):
            blc.filter_confident_candidates(
                topk_tokens=[[1]],
                topk_probs=[[0.5], [0.5]],
                prob_threshold=0.1,
            )


class TestNgramWorkerWiring(unittest.TestCase):
    """Prove the worker (a non-new module) actually invokes the new capability."""

    def test_worker_wires_bonus_logit_seeds(self):
        pytest.importorskip("torch", reason="requires torch")
        pytest.importorskip("sgl_kernel", reason="requires sgl_kernel")
        from sglang.srt.speculative import ngram_worker

        # The worker imports the seeding entry point and exposes the hook that
        # calls it after verify — this is the integration under test.
        self.assertIs(
            ngram_worker.bonus_logit_corpus_seeds, blc.bonus_logit_corpus_seeds
        )
        self.assertTrue(
            hasattr(ngram_worker.NGRAMWorker, "_seed_corpus_from_bonus_logits")
        )


if __name__ == "__main__":
    unittest.main()
