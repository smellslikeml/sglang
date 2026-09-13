"""Confidence-thresholded dynamic draft length.

Adapted from *Improving Multi-candidate Speculative Decoding* (DynaSD,
arXiv:2409.10644). DynaSD observes that extending a draft chain is only worth
the compute while the *cumulative* probability that the whole chain survives
verification stays high; once that joint confidence collapses, the extra draft
tokens are almost always rejected. Rather than always drafting a fixed number
of steps, it stops as soon as the running confidence product falls below a
threshold, yielding a data-dependent draft length.

Adaptation (Mode 2): the dynamic-length rule is ported at full fidelity. The
auxiliary signal is substituted with a target-native, parameter-free proxy.
DynaSD reads confidence from the draft model's softmax at draft time; plumbing
that raw distribution out of SGLang's CUDA-graph draft loop is invasive and
device-synchronous. Instead we reconstruct an equivalent per-position
confidence from the empirical acceptance profile already produced at verify
time: the fraction of requests whose draft was still accepted at each position.
Smoothed with an EMA, this profile is exactly the marginal per-position
acceptance probability DynaSD's rule consumes.

The resulting confidence-implied length is used by ``AdaptiveStepSlot`` as an
upward ceiling on ``speculative_num_steps`` — a draft-length signal that
complements the existing accept-length EMA step controller.
"""

from __future__ import annotations

from typing import List, Sequence


def accept_profile(
    num_correct_drafts_per_req: Sequence[int], num_positions: int
) -> List[float]:
    """Empirical per-position marginal acceptance probability.

    ``accept_profile(...)[pos]`` is the fraction of requests whose verified
    draft reached at least position ``pos + 1`` — i.e. the probability that the
    draft token at position ``pos`` was accepted, given the whole batch.

    Args:
        num_correct_drafts_per_req: Per-request accepted draft counts (drafts
            only, no bonus token), as produced at verify time.
        num_positions: Number of draft positions to report (typically the
            largest candidate step count).
    """
    if num_positions <= 0:
        return []
    n = len(num_correct_drafts_per_req)
    if n == 0:
        return [0.0] * num_positions
    return [
        sum(1 for c in num_correct_drafts_per_req if c > pos) / n
        for pos in range(num_positions)
    ]


def confidence_implied_length(profile: Sequence[float], *, threshold: float) -> int:
    """DynaSD dynamic-length rule.

    Returns the number of leading draft positions whose cumulative acceptance
    confidence (the running product of per-position probabilities) stays at or
    above ``threshold``. A higher threshold demands more confidence to keep
    extending and therefore yields a shorter draft; ``threshold <= 0`` never
    truncates.
    """
    length = 0
    cumulative = 1.0
    for p in profile:
        cumulative *= p
        if cumulative < threshold:
            break
        length += 1
    return length


class DraftConfidenceTracker:
    """EMA over the per-position acceptance profile, exposing DynaSD's length.

    The tracker starts optimistic (every position at probability 1.0) so it
    imposes no ceiling until enough verify results have pulled the tail down —
    mirroring the warmup discipline of the accept-length EMA it complements.
    """

    def __init__(self, num_positions: int, ema_alpha: float = 0.2):
        assert num_positions >= 0, "num_positions must be non-negative"
        self.num_positions = num_positions
        self.ema_alpha = ema_alpha
        self.profile: List[float] = [1.0] * num_positions

    def update(self, num_correct_drafts_per_req: Sequence[int]) -> None:
        """Fold one batch of verify results into the EMA profile."""
        if not num_correct_drafts_per_req or self.num_positions == 0:
            return
        observed = accept_profile(num_correct_drafts_per_req, self.num_positions)
        alpha = self.ema_alpha
        self.profile = [
            (1.0 - alpha) * old + alpha * new
            for old, new in zip(self.profile, observed)
        ]

    def implied_length(self, threshold: float) -> int:
        """Confidence-implied draft length under the current EMA profile."""
        return confidence_implied_length(self.profile, threshold=threshold)
