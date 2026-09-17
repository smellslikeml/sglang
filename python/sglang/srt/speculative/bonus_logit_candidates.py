"""Final-layer bonus-logit candidate supplementation for draft-model-free
speculative decoding.

Adapted from the outer-loop idea in *ECHO: Early-layer Collaborative
Hierarchical Orchestration with Bonus Logits in Speculative Decoding*
(https://arxiv.org/abs/2609.17241). ECHO observes that the target model's
authoritative final-layer distribution — the "bonus logits" already produced at
the last accepted position of every verify step — is a free, high-confidence
signal about what the model wants to generate next. Its outer loop uses that
signal to *supplement the draft tree with high-confidence candidates for
subsequent cycles*.

This module ports only that insight, target-native, for the NGRAM worker: the
"draft tree for subsequent cycles" is the n-gram corpus, so the high-confidence
bonus-logit tokens are folded back into the corpus as short
``recent-context -> high-confidence-token`` seeds. Under greedy decoding only
the argmax token is ever realized and inserted, so the corpus never learns the
model's *other* confident continuations; seeding them lets the next draft round
propose branches the string matcher would otherwise miss.

Everything here is parameter-free and stdlib-only: the caller does the (cheap,
on-device) softmax + top-k extraction and hands the tiny per-request result to
these pure helpers. ECHO's early-layer inner loop, its state-reuse verification,
and its one-shot fine-tuning dependency are intentionally out of scope.
"""

from __future__ import annotations

from typing import List, Sequence

# Defaults chosen to be conservative: a small top-k keeps the extra corpus
# inserts cheap, and the probability floor keeps only genuinely confident
# alternatives out of the tree (low-probability tail tokens are noise, not
# candidates worth speculating on).
DEFAULT_BONUS_LOGIT_TOP_K: int = 3
DEFAULT_BONUS_LOGIT_PROB_THRESHOLD: float = 0.1


def filter_confident_candidates(
    *,
    topk_tokens: Sequence[Sequence[int]],
    topk_probs: Sequence[Sequence[float]],
    prob_threshold: float,
) -> List[List[int]]:
    """Keep, per request, the bonus-logit token ids whose probability clears
    ``prob_threshold``.

    ``topk_tokens`` / ``topk_probs`` are the already-extracted top-k token ids
    and their softmax probabilities (one row per request, highest first).
    Returns one de-duplicated list of confident token ids per request; a request
    with no token above the floor yields an empty list.
    """
    if len(topk_tokens) != len(topk_probs):
        raise ValueError(
            f"topk_tokens ({len(topk_tokens)}) and topk_probs "
            f"({len(topk_probs)}) must describe the same requests."
        )

    confident: List[List[int]] = []
    for tokens, probs in zip(topk_tokens, topk_probs):
        seen: set = set()
        kept: List[int] = []
        for token, prob in zip(tokens, probs):
            if prob < prob_threshold:
                # Rows are sorted high-to-low, so the first miss ends the row.
                break
            if token not in seen:
                seen.add(token)
                kept.append(int(token))
        confident.append(kept)
    return confident


def build_corpus_seed_sequences(
    *,
    context_tails: Sequence[Sequence[int]],
    candidate_tokens: Sequence[Sequence[int]],
    max_trie_depth: int,
) -> List[List[int]]:
    """Build ``recent-context + high-confidence-token`` seed sequences to insert
    into the n-gram corpus.

    For each request, every confident candidate token is appended to that
    request's recent context tail, trimmed so the seed never exceeds
    ``max_trie_depth`` tokens (the deepest suffix the corpus indexes). Requests
    with no confident candidate contribute nothing.
    """
    if len(context_tails) != len(candidate_tokens):
        raise ValueError(
            f"context_tails ({len(context_tails)}) and candidate_tokens "
            f"({len(candidate_tokens)}) must describe the same requests."
        )
    if max_trie_depth < 1:
        raise ValueError(f"max_trie_depth must be >= 1, got {max_trie_depth}.")

    prefix_budget = max_trie_depth - 1
    seeds: List[List[int]] = []
    for tail, tokens in zip(context_tails, candidate_tokens):
        prefix = list(tail[-prefix_budget:]) if prefix_budget else []
        for token in tokens:
            seeds.append(prefix + [int(token)])
    return seeds


def bonus_logit_corpus_seeds(
    *,
    topk_tokens: Sequence[Sequence[int]],
    topk_probs: Sequence[Sequence[float]],
    context_tails: Sequence[Sequence[int]],
    prob_threshold: float,
    max_trie_depth: int,
) -> List[List[int]]:
    """End-to-end: turn per-request bonus-logit top-k into corpus seed
    sequences.

    Thin composition of :func:`filter_confident_candidates` and
    :func:`build_corpus_seed_sequences`; the single entry point the NGRAM worker
    calls after a verify step.
    """
    candidate_tokens = filter_confident_candidates(
        topk_tokens=topk_tokens,
        topk_probs=topk_probs,
        prob_threshold=prob_threshold,
    )
    return build_corpus_seed_sequences(
        context_tails=context_tails,
        candidate_tokens=candidate_tokens,
        max_trie_depth=max_trie_depth,
    )
