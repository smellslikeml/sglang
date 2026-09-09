"""Block-sparse prefill attention (mean-corrected).

This module implements the algorithmic core of *FlashPrefill V2: Block-Sparse
Prefill Attention for Long-Context LLM Serving* (https://arxiv.org/abs/2608.19758)
as a target-native attention backend for SGLang.

What is ported (full fidelity):
  * Block-level pattern discovery scored by each KV block's representative
    (max) logit against the query.
  * Max-based dynamic thresholding: a KV block is attended to only when its
    representative logit lies within ``keep_threshold`` of the per-query peak,
    so the retained set adapts to the discovered attention pattern.
  * The V2 *mean correction term*: rather than hard-dropping unselected blocks
    (which over-normalizes the surviving softmax weights), the aggregate
    contribution of each dropped block is estimated from its *mean* logit and
    *mean* value and folded back into the softmax numerator/denominator. This
    is the paper's key accuracy fix that keeps degradation manageable at
    extreme sparsity.

What is substituted (auxiliary, per Mode-2 adaptation): the paper's bespoke
FlashAttention-3/4-aligned CUDA kernel (PackGQA memory access, warp
specialization, pingpong pipelining, FP8) is replaced by SGLang's existing
torch-native paged-KV gather + dense math. The speedups the paper reports come
from that kernel; here we deliver the *approximation quality* of the V2
selection + correction so it can be exercised and validated inside SGLang's
continuous-batching prefill path on any device. A fused sparse kernel is a
downstream optimization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

# Defaults tuned for long-context prefill. ``keep_threshold`` is expressed in
# (scaled) logit space: a block whose max logit is more than this below the
# per-query peak contributes < exp(-keep_threshold) relative softmax mass and
# is a candidate for the cheap mean-corrected estimate instead of exact math.
DEFAULT_BLOCK_SIZE = 128
DEFAULT_KEEP_THRESHOLD = 8.0
DEFAULT_MIN_BLOCKS = 1
# Below this KV length sparsity buys nothing, so we fall back to exact attention.
DEFAULT_MIN_CONTEXT = 4 * DEFAULT_BLOCK_SIZE


def block_sparse_prefill_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    causal: bool = True,
    block_size: int = DEFAULT_BLOCK_SIZE,
    keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    mean_correction: bool = True,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Mean-corrected block-sparse attention for a single sequence.

    Args:
        query: ``[num_heads, q_len, head_dim]``.
        key:   ``[num_kv_heads, kv_len, head_dim]``.
        value: ``[num_kv_heads, kv_len, v_head_dim]``.
        scaling: softmax logit scale (``1/sqrt(head_dim)`` typically).
        causal: apply causal masking (query ``i`` aligned to the tail of ``kv``).
        block_size: KV block granularity for pattern discovery.
        keep_threshold: max-based dynamic threshold, in scaled-logit units. Use
            ``float("inf")`` to keep every block (exact dense attention).
        min_blocks: always retain at least this many top-scoring blocks per query.
        mean_correction: fold the V2 mean-correction estimate of dropped blocks
            back into the softmax. When ``False`` this degrades to plain
            hard-drop block-sparse attention (the pre-V2 behavior).
        enable_gqa: broadcast KV heads across grouped query heads.

    Returns:
        ``[num_heads, q_len, v_head_dim]`` attention output, in ``query.dtype``.
    """
    num_heads, q_len, _ = query.shape
    num_kv_heads = key.shape[0]
    if enable_gqa and num_kv_heads != num_heads:
        group = num_heads // num_kv_heads
        key = key.repeat_interleave(group, dim=0)
        value = value.repeat_interleave(group, dim=0)

    kv_len = key.shape[1]
    out_dtype = query.dtype
    qf = query.float()
    kf = key.float()
    vf = value.float()

    logits = torch.einsum("hqd,hkd->hqk", qf, kf) * scaling  # [H, q, kv]

    if causal:
        q_pos = torch.arange(q_len, device=query.device) + (kv_len - q_len)
        k_pos = torch.arange(kv_len, device=query.device)
        allowed = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # [q, kv]
        logits = logits.masked_fill(~allowed.unsqueeze(0), float("-inf"))

    # Pad the KV axis up to a whole number of blocks so we can reshape.
    n_blocks = (kv_len + block_size - 1) // block_size
    pad = n_blocks * block_size - kv_len
    if pad:
        logits = torch.nn.functional.pad(logits, (0, pad), value=float("-inf"))
    logits_blk = logits.view(num_heads, q_len, n_blocks, block_size)

    valid = torch.isfinite(logits_blk)  # [H, q, n_blocks, block_size]
    counts = valid.sum(dim=-1)  # keys attended per (head, query, block)
    block_max = logits_blk.amax(dim=-1)  # representative (max) logit per block
    row_max = block_max.amax(dim=-1, keepdim=True)  # per-query peak == global max

    # --- Max-based dynamic thresholding -----------------------------------
    keep = block_max >= (row_max - keep_threshold)
    keep &= counts > 0  # never "keep" a fully-masked block
    if min_blocks > 1:
        # Guarantee a floor of top-scoring blocks even under a tight threshold.
        k = min(min_blocks, n_blocks)
        topk = block_max.topk(k, dim=-1).indices
        keep.scatter_(-1, topk, counts.gather(-1, topk) > 0)

    m = row_max.clamp(min=-1e30)  # softmax stabilizer, guards all-masked rows
    exp_logits = torch.exp(logits_blk - m.unsqueeze(-1))
    exp_logits = torch.where(valid, exp_logits, torch.zeros_like(exp_logits))

    # --- Exact contribution of the retained blocks ------------------------
    kept_token_mask = keep.unsqueeze(-1) & valid  # [H, q, n_blocks, block_size]
    exp_kept = torch.where(
        kept_token_mask, exp_logits, torch.zeros_like(exp_logits)
    ).view(num_heads, q_len, n_blocks * block_size)
    denom = exp_kept.sum(dim=-1)  # [H, q]
    numer = torch.einsum("hqk,hkd->hqd", exp_kept, _pad_value(vf, pad))

    # --- V2 mean correction for the dropped blocks ------------------------
    if mean_correction:
        safe_counts = counts.clamp(min=1)
        summed = torch.where(
            valid, logits_blk, torch.zeros_like(logits_blk)
        ).sum(dim=-1)
        mean_logit = summed / safe_counts  # [H, q, n_blocks]
        # Estimated softmax mass of a dropped block: (#keys) * exp(mean - max).
        block_weight = counts * torch.exp(mean_logit - m)
        dropped = (~keep) & (counts > 0)
        block_weight = torch.where(
            dropped, block_weight, torch.zeros_like(block_weight)
        )
        denom = denom + block_weight.sum(dim=-1)
        mean_value = _block_mean_value(vf, pad, n_blocks, block_size)  # [H, nb, d]
        numer = numer + torch.einsum("hqb,hbd->hqd", block_weight, mean_value)

    out = numer / denom.clamp(min=1e-20).unsqueeze(-1)
    return out.to(out_dtype)


def _pad_value(value: torch.Tensor, pad: int) -> torch.Tensor:
    """Right-pad the KV axis of ``value`` with zeros (padded keys carry no mass)."""
    if not pad:
        return value
    return torch.nn.functional.pad(value, (0, 0, 0, pad))


def _block_mean_value(
    value: torch.Tensor, pad: int, n_blocks: int, block_size: int
) -> torch.Tensor:
    """Per-block mean value vector, ignoring right-padding. ``[H, n_blocks, d]``."""
    num_heads, kv_len, v_dim = value.shape
    padded = _pad_value(value, pad)  # [H, n_blocks*block_size, d]
    blocks = padded.view(num_heads, n_blocks, block_size, v_dim)
    real = torch.zeros(n_blocks * block_size, device=value.device)
    real[:kv_len] = 1.0
    real = real.view(1, n_blocks, block_size, 1)
    counts = real.sum(dim=2).clamp(min=1.0)  # [1, n_blocks, 1]
    return (blocks * real).sum(dim=2) / counts


class BlockSparsePrefillAttnBackend(TorchNativeAttnBackend):
    """Torch-native attention backend that runs mean-corrected block-sparse
    attention on the long-context extend (prefill) path.

    Decode, cross-attention, sliding-window, and short-context prefill all defer
    to the exact :class:`TorchNativeAttnBackend` implementation; only causal
    prefill over a long enough context takes the sparse route.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        keep_threshold: float = DEFAULT_KEEP_THRESHOLD,
        min_blocks: int = DEFAULT_MIN_BLOCKS,
        min_context: int = DEFAULT_MIN_CONTEXT,
        mean_correction: bool = True,
    ):
        super().__init__(model_runner)
        self.block_size = block_size
        self.keep_threshold = keep_threshold
        self.min_blocks = min_blocks
        self.min_context = min_context
        self.mean_correction = mean_correction

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor] = None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
        sliding_window_size: Optional[int] = None,
    ):
        sliding = sliding_window_size is not None and sliding_window_size > -1
        if not causal or is_cross_attn or sliding:
            # Patterns this backend does not sparsify -> exact torch-native path.
            return super()._run_sdpa_forward_extend(
                query,
                output,
                k_cache,
                v_cache,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_prefix_lens,
                extend_seq_lens,
                encoder_lens,
                scaling=scaling,
                enable_gqa=enable_gqa,
                causal=causal,
                is_cross_attn=is_cross_attn,
                sliding_window_size=sliding_window_size,
            )

        # [num_tokens, num_heads, head] -> [num_heads, num_tokens, head]
        query = query.movedim(0, query.dim() - 2)
        start_q = 0
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len_q = extend_seq_lens[seq_idx]
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q

            per_req_query = query[:, start_q:end_q, :]
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)
            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            # Short contexts gain nothing from sparsity: keep every block so the
            # result is exact dense attention.
            keep_threshold = (
                self.keep_threshold
                if seq_len_kv >= self.min_context
                else float("inf")
            )
            per_req_out = block_sparse_prefill_attention(
                per_req_query,
                per_req_key,
                per_req_value,
                scaling=scaling,
                causal=True,
                block_size=self.block_size,
                keep_threshold=keep_threshold,
                min_blocks=self.min_blocks,
                mean_correction=self.mean_correction,
                enable_gqa=enable_gqa,
            )
            output[start_q:end_q, :, :] = per_req_out.movedim(query.dim() - 2, 0)
            start_q = end_q
        return output
