"""Recycle verify-tree draft branches back into the n-gram candidate pool.

Adapted from *Ouroboros: Generating Longer Drafts Phrase by Phrase for Faster
Speculative Decoding* (https://arxiv.org/abs/2402.13720). Ouroboros keeps a
candidate-phrase pool that is continuously refined across decoding steps: the
phrases a step generates are fed back so later steps can draft longer
continuations at low cost — the "snake eating its tail" that names the method.

SGLang's ``NGRAMWorker`` already hosts such a pool (the ``NgramCorpus``), but it
only recycles the *linear* accepted sequence (the trailing ``origin_input_ids +
output_ids`` window). The multi-branch draft *tree* that each verify step
proposes — including the sibling branches the target did not take this step — is
discarded. This module captures Ouroboros' recycling idea target-natively: it
turns those proposed leaf-path branches into anchored token phrases that can be
re-inserted into the corpus, so a recurring context can retrieve a longer draft
immediately.

Correctness is unaffected: every draft the corpus proposes is still verified by
the target model, so recycled-but-wrong phrases only cost draft slots, never
output quality. Because they *can* dilute the pool's hit rate, recycling is
opt-in (``--speculative-ngram-recycle-draft-phrases``).

This is a pure, host-side policy: it takes already-extracted leaf paths plus each
request's context window and returns the token sequences to feed to
``NgramCorpus.batch_put``. Leaf-path extraction itself is reused from
``NgramCorpus.leaf_paths_from_mask`` by the caller.
"""

from __future__ import annotations

from typing import List, Sequence

# Padding value the corpus uses to fill unmatched draft slots; a leaf path is
# truncated at the first occurrence so padding never enters a recycled phrase.
_PAD_TOKEN = 0


def _continuation_after_anchor(path: Sequence[int]) -> List[int]:
    """Return the phrase continuation of ``path`` (everything after the anchor
    root), stopping at the first padding token.

    A leaf path from ``leaf_paths_from_mask`` starts at the tree root, which is
    the last context token (the anchor). The tokens after it are the proposed
    continuation; unmatched tree slots are ``_PAD_TOKEN`` and terminate it.
    """
    continuation: List[int] = []
    for token in path[1:]:
        if token == _PAD_TOKEN:
            break
        continuation.append(int(token))
    return continuation


def build_recycled_phrases(
    *,
    leaf_paths: List[List[List[int]]],
    context_tails: List[List[int]],
    min_phrase_len: int = 2,
    max_phrases_per_req: int = 4,
) -> List[List[int]]:
    """Build the recycled candidate phrases to insert back into the corpus.

    Args:
        leaf_paths: Per request, the draft tree's leaf paths (each path starts
            with the context anchor token, matching
            ``NgramCorpus.leaf_paths_from_mask``).
        context_tails: Per request, the recent context window ending at the
            anchor token (the same window the caller inserts linearly). Used to
            key the recycled continuation so future contexts can retrieve it.
        min_phrase_len: Minimum continuation length (tokens beyond the anchor);
            shorter branches carry no multi-token phrase value and are skipped.
        max_phrases_per_req: Cap on recycled phrases per request, longest first,
            to bound the extra insert cost.

    Returns:
        A flat list of ``context_tail[:-1] + leaf_path`` token sequences across
        the batch, de-duplicated per request. Empty when nothing qualifies.
    """
    if len(leaf_paths) != len(context_tails):
        raise ValueError(
            f"leaf_paths ({len(leaf_paths)}) and context_tails "
            f"({len(context_tails)}) must be per-request aligned."
        )

    recycled: List[List[int]] = []
    for req_paths, context_tail in zip(leaf_paths, context_tails):
        if not context_tail:
            continue
        # context_tail ends with the anchor token, which is also each leaf
        # path's root; appending the continuation reconstructs the linear
        # "context then proposed continuation" sequence the pool keys on.
        prefix = list(context_tail)

        seen: set[tuple[int, ...]] = set()
        req_phrases: List[List[int]] = []
        for path in req_paths:
            if not path:
                continue
            continuation = _continuation_after_anchor(path)
            if len(continuation) < min_phrase_len:
                continue
            phrase = prefix + continuation
            key = tuple(phrase)
            if key in seen:
                continue
            seen.add(key)
            req_phrases.append(phrase)

        # Longest continuations first: they extend future drafts the most.
        req_phrases.sort(key=len, reverse=True)
        recycled.extend(req_phrases[:max_phrases_per_req])

    return recycled
