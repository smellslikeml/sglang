"""Tests for Ouroboros-style draft-phrase recycling (arXiv:2402.13720).

The pure-policy cases pin down the anchoring / filtering contract of
``build_recycled_phrases``. The round-trip case wires that output through the
real (non-new) ``NgramCorpus`` — the same pool the NGRAM worker recycles into —
to prove a recycled branch is corpus-ingestible and makes a previously barren
context productive.
"""

import unittest

import numpy as np

from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.phrase_recycler import build_recycled_phrases
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestBuildRecycledPhrases(CustomTestCase):
    """Derived-property tests for the recycling policy."""

    def test_anchors_continuation_onto_context(self):
        # Leaf path root (5) is the anchor and equals context_tail[-1]; the
        # recycled sequence must be context_tail + the continuation after it.
        phrases = build_recycled_phrases(
            leaf_paths=[[[5, 6, 7]]],
            context_tails=[[2, 3, 4, 5]],
        )
        self.assertEqual(phrases, [[2, 3, 4, 5, 6, 7]])

    def test_short_branches_are_skipped(self):
        # A single-token continuation carries no multi-token phrase value.
        phrases = build_recycled_phrases(
            leaf_paths=[[[5, 6]]],
            context_tails=[[4, 5]],
            min_phrase_len=2,
        )
        self.assertEqual(phrases, [])

    def test_padding_truncates_continuation(self):
        # Unmatched draft slots are 0-padded; they must not enter a phrase, and
        # a branch that falls below min_phrase_len after truncation is dropped.
        phrases = build_recycled_phrases(
            leaf_paths=[[[5, 6, 7, 0, 0], [5, 8, 0, 0, 0]]],
            context_tails=[[5]],
            min_phrase_len=2,
        )
        self.assertEqual(phrases, [[5, 6, 7]])

    def test_dedup_and_longest_first_cap(self):
        # Duplicate branches collapse; the cap keeps the longest continuations.
        phrases = build_recycled_phrases(
            leaf_paths=[[[9, 1, 2, 3], [9, 1, 2, 3], [9, 4], [9, 5, 6]]],
            context_tails=[[8, 9]],
            min_phrase_len=1,
            max_phrases_per_req=2,
        )
        self.assertEqual(phrases, [[8, 9, 1, 2, 3], [8, 9, 5, 6]])

    def test_misaligned_inputs_raise(self):
        with self.assertRaises(ValueError):
            build_recycled_phrases(leaf_paths=[[[1, 2]]], context_tails=[])

    def test_empty_context_tail_skipped(self):
        phrases = build_recycled_phrases(
            leaf_paths=[[[5, 6, 7]]],
            context_tails=[[]],
        )
        self.assertEqual(phrases, [])


class TestPhraseRecyclingRoundTrip(CustomTestCase):
    """Recycled branches must round-trip through the real corpus pool."""

    def test_recycled_branch_enriches_pool(self):
        draft = 8
        # Seed a corpus so a real query yields a multi-token draft tree.
        source = NgramCorpus(max_trie_depth=12, draft_token_num=draft)
        source.batch_put([[100, 200, 3, 41, 42, 43, 44]])
        source.synchronize()
        ids, masks = source.batch_get(
            req_ids=["q"],
            batch_tokens=[[100, 200, 3]],
            total_lens=[3],
        )
        ids = ids.reshape(-1, draft)[0].tolist()
        masks = masks.reshape(-1, draft, draft)[0].tolist()
        self.assertEqual(ids[0], 3, "anchor should be the last context token")

        leaf_paths = source.leaf_paths_from_mask(ids, masks)

        # Recycle the proposed branch onto a *novel* context that shares only the
        # anchor token (3). The prefix 71,72,73 was never seen by any pool.
        novel_context = [71, 72, 73, 3]
        recycled = build_recycled_phrases(
            leaf_paths=[leaf_paths],
            context_tails=[novel_context],
        )
        self.assertTrue(recycled, "expected at least one recycled phrase")
        # Every recycled phrase keeps the novel context prefix and extends it.
        for phrase in recycled:
            self.assertEqual(phrase[: len(novel_context)], novel_context)
            self.assertGreater(len(phrase), len(novel_context))

        # A fresh pool holding only the recycled phrase must now draft the
        # continuation for the novel context that had no prior match.
        pool = NgramCorpus(max_trie_depth=12, draft_token_num=draft)
        before, _ = pool.batch_get(
            req_ids=["novel-before"],
            batch_tokens=[novel_context],
            total_lens=[len(novel_context)],
        )
        before = before.reshape(-1, draft)[0].tolist()
        self.assertTrue(
            all(t == 0 for t in before[1:]),
            f"novel context should have no draft before recycling, got {before}",
        )

        pool.batch_put(recycled)
        pool.synchronize()
        after, _ = pool.batch_get(
            req_ids=["novel-after"],
            batch_tokens=[novel_context],
            total_lens=[len(novel_context)],
        )
        after = after.reshape(-1, draft)[0].tolist()
        recycled_continuation = recycled[0][len(novel_context) :]
        self.assertIn(
            recycled_continuation[0],
            after,
            f"recycled continuation should be draftable after recycling, got {after}",
        )
        np.testing.assert_array_equal(after[0], 3)


if __name__ == "__main__":
    unittest.main(verbosity=3)
