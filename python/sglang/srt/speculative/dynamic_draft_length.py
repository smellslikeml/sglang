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

That empirical profile is an *unconditional* survival curve — position ``pos``
holds ``P(accept_length > pos)`` — whereas DynaSD's running product consumes the
*conditional* acceptance ``p_pos = P(accept pos | accepted 0..pos-1)``. We
convert survival to conditional (``conditional_accept_profile``) before applying
the rule, so the running product telescopes back to the true joint survival of
the whole chain rather than a product of survivals that over-truncates long
drafts. Smoothed with an EMA, this is the marginal signal DynaSD's rule expects.

The resulting confidence-implied length is used by ``AdaptiveStepSlot`` as an
upward ceiling on ``speculative_num_steps`` — a draft-length signal that
complements the existing accept-length EMA step controller.
"""

from __future__ import annotations

from typing import List, Sequence


def accept_profile(
    num_correct_drafts_per_req: Sequence[int], num_positions: int
) -> List[float]:
    """Empirical per-position *unconditional* acceptance probability.

    ``accept_profile(...)[pos]`` is the fraction of requests whose verified
    draft reached at least position ``pos + 1`` — i.e. the survival probability
    ``P(accept_length > pos)`` across the batch. The curve is monotonically
    non-increasing; ``conditional_accept_profile`` turns it into the per-step
    conditional acceptance the running-product rule consumes.

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


def conditional_accept_profile(profile: Sequence[float]) -> List[float]:
    """Per-step *conditional* acceptance derived from an unconditional profile.

    ``accept_profile`` reports the unconditional survival ``S_pos`` that a draft
    reaches each position. DynaSD's running-product rule instead needs the
    conditional acceptance ``p_pos = P(accept pos | accepted 0..pos-1) =
    S_pos / S_{pos-1}`` (with ``S_{-1} = 1``). Multiplying these conditionals
    telescopes back to the joint survival ``S_{L-1}`` of a length-``L`` chain, so
    ``confidence_implied_length`` measures the true probability the whole chain
    clears verification instead of a product of survivals that under-counts it.
    A zero survival makes every later conditional zero (the chain is already
    dead), which the rule reads as an immediate stop.
    """
    conditional: List[float] = []
    prev = 1.0
    for p in profile:
        conditional.append(p / prev if prev > 0.0 else 0.0)
        prev = p
    return conditional


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
        """Confidence-implied draft length under the current EMA profile.

        The EMA tracks an unconditional survival curve; it is converted to
        per-step conditional acceptance so the running product recovers the true
        joint chain survival (see ``conditional_accept_profile``).
        """
        conditional = conditional_accept_profile(self.profile)
        return confidence_implied_length(conditional, threshold=threshold)
