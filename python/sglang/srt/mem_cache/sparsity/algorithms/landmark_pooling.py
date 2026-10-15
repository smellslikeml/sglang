"""Fine-grained multi-landmark page pooling for Hierarchical Sparse Attention.

Adapted from "Random Long-Context Access for Mamba via Hardware-aligned
Hierarchical Sparse Attention" (https://arxiv.org/abs/2504.16795).

The paper's core innovation is learning *token-to-chunk relevance based on
fine-grained token-level information inside each chunk*: a chunk that holds a
single strongly-matching token should still be selected even when the rest of
the chunk is unrelated. A single masked-mean landmark per page averages that
signal away, so a needle token flanked by low-relevance tokens can drop out of
the top-k.

These helpers provide a parameter-free proxy for that fine-grained signal
(no learned relevance head, keeping the selector inference-only). Each page's
tokens are split into ``num_landmarks`` contiguous sub-segments, and each
sub-segment is masked-mean pooled into its own landmark key. At retrieval time
a page's relevance is the *maximum* scaled query-landmark dot product over its
landmarks, so one strongly matching sub-segment lifts the whole page into the
top-k. With ``num_landmarks == 1`` this reduces exactly to the original
mean-landmark baseline, so the extension is opt-in and backward compatible.
"""

import torch


def landmark_segment_ids(page_size: int, num_landmarks: int, device) -> torch.Tensor:
    """Map each intra-page token position to its landmark sub-segment.

    Positions are split into ``num_landmarks`` contiguous, near-equal segments.
    Returns an int64 tensor of shape ``[page_size]`` with values in
    ``[0, num_landmarks)``.
    """
    positions = torch.arange(page_size, device=device)
    return (positions * num_landmarks) // page_size


def pool_page_landmarks(
    keys: torch.Tensor,
    tok_mask: torch.Tensor,
    num_landmarks: int,
) -> torch.Tensor:
    """Pool per-token keys into ``num_landmarks`` landmark keys per page.

    Args:
        keys: ``[n, max_pages, page_size, head_num, head_dim]`` float key vectors.
        tok_mask: ``[n, max_pages, page_size]`` bool; True for real (in-range) tokens.
        num_landmarks: number of sub-segment landmarks per page (>= 1).

    Returns:
        ``[n, max_pages, num_landmarks, head_num, head_dim]`` landmark keys.
        A landmark with no valid tokens is left as zeros (it contributes a
        relevance of 0 and is dominated by the max over real landmarks).
    """
    if num_landmarks < 1:
        raise ValueError(f"num_landmarks must be >= 1, got {num_landmarks}")

    page_size = keys.shape[2]
    seg_ids = landmark_segment_ids(page_size, num_landmarks, keys.device)

    # One-hot [page_size, num_landmarks] assignment of tokens to landmarks.
    onehot = torch.zeros(
        (page_size, num_landmarks), dtype=keys.dtype, device=keys.device
    )
    onehot[torch.arange(page_size, device=keys.device), seg_ids] = 1.0

    # Zero out padded tokens so they contribute nothing to their segment.
    weights = tok_mask.to(keys.dtype).unsqueeze(-1) * onehot  # [n, mp, ps, L]

    # Sum keys per landmark, then divide by the per-landmark valid-token count.
    sums = torch.einsum("nmpl,nmphd->nmlhd", weights, keys)
    counts = weights.sum(dim=2).clamp(min=1.0)  # [n, mp, L]
    return sums / counts.unsqueeze(-1).unsqueeze(-1)


def score_pages_by_landmarks(
    query: torch.Tensor,
    landmarks: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Score pages by the max scaled query-landmark dot product.

    Args:
        query: ``[bs, kv_heads, head_dim]`` query aligned to the KV heads.
        landmarks: ``[bs, num_pages, num_landmarks, kv_heads, head_dim]``.
        scale: attention scaling factor (typically ``1 / sqrt(head_dim)``).

    Returns:
        ``[bs, num_pages]`` page relevance = max over landmarks of the summed,
        scaled query-landmark dot product across heads and dims.
    """
    q = query.unsqueeze(1).unsqueeze(1)  # [bs, 1, 1, kv_heads, head_dim]
    per_landmark = (q * landmarks).sum(dim=(3, 4)) * scale  # [bs, num_pages, L]
    return per_landmark.max(dim=2).values
