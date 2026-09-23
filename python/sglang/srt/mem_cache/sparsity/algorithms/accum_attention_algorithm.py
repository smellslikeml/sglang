"""Accumulated-attention sparse retrieval.

Adapted from ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient
Long-Context LLMs (https://arxiv.org/abs/2503.10714).

ZSMerge's core, training-free insight is that a cached token's importance is not
captured by its relevance to the *current* query alone (as in Quest's bounding-box
criticality), but by the attention it has *accumulated* across the generation so
far. Heavy-hitter pages that were consistently attended to remain important even
when the current query does not peak on them.

This module keeps that mechanism at full fidelity: each decode step it maintains a
decayed running sum of per-page attention affinity and retains the top-k pages by
that accumulated importance (recent-page retention is provided by the base class).

Mode-2 (adapted port) substitutions, made because the sparse-retrieval contract
here selects KV indices *before* attention runs and never owns the KV pool:

  * ZSMerge's residual feature-*merging* step (evicted tokens are fused into
    survivors) is dropped in favor of plain top-k retention. Merging would require
    mutating the shared KV pool, which this layer only reads from.
  * True post-softmax per-token attention weights are unavailable pre-attention, so
    the accumulated signal is approximated by query x page-centroid affinity passed
    through a softmax over pages -- the same page-representation shortcut Quest uses,
    reused here as the per-step attention proxy that feeds the accumulator.
"""

import logging

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)

logger = logging.getLogger(__name__)


class AccumAttentionAlgorithm(BaseSparseAlgorithmImpl):
    """Page-wise sparse retrieval driven by decayed accumulated attention."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        extra = config.sparse_extra_config
        # Decay applied to the running importance before adding the current step.
        # 1.0 => pure H2O-style running sum; <1.0 => exponential recency emphasis.
        self.accum_decay = float(extra.get("accum_decay", 0.9))
        if not 0.0 < self.accum_decay <= 1.0:
            raise ValueError(f"accum_decay must be in (0, 1], got {self.accum_decay}")
        self.page_key_mean = {}
        self.page_valid = {}
        # Per-layer [num_pages] decayed accumulated attention importance.
        self.page_attn_accum = {}

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)
        head_num, head_dim = key_buf.shape[1], key_buf.shape[2]

        for layer_id in range(start_layer, end_layer):
            self.page_key_mean[layer_id] = torch.zeros(
                (total_num_pages, head_num, head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.page_valid[layer_id] = torch.zeros(
                total_num_pages, dtype=torch.bool, device=self.device
            )
            self.page_attn_accum[layer_id] = torch.zeros(
                total_num_pages, dtype=torch.float32, device=self.device
            )

        logger.info(
            "Initialized accumulated-attention page reps: %d pages, %d layers, "
            "head_num=%d, head_dim=%d, accum_decay=%.3f",
            total_num_pages,
            end_layer - start_layer,
            head_num,
            head_dim,
            self.accum_decay,
        )

    def _compute_page_representations(
        self,
        layer_id: int,
        reqs: torch.Tensor,
        seq_lens: torch.Tensor,
        start_page,
        end_page: torch.Tensor,
        k_buffer: torch.Tensor,
    ):
        if isinstance(start_page, int):
            start_page = torch.full_like(end_page, start_page)

        device = k_buffer.device
        req_to_token = self.req_to_token_pool.req_to_token
        n = reqs.shape[0]
        max_pages = int((end_page - start_page).max().item())
        if max_pages <= 0:
            return

        pg_off = torch.arange(max_pages, device=device).unsqueeze(0)
        pg_id = start_page.unsqueeze(1) + pg_off
        pg_mask = pg_id < end_page.unsqueeze(1)

        tok_start = pg_id * self.page_size
        tok_off = torch.arange(self.page_size, device=device).view(1, 1, -1)
        tok_pos = tok_start.unsqueeze(2) + tok_off
        tok_mask = (
            tok_pos
            < (tok_start + self.page_size).clamp(max=seq_lens.unsqueeze(1)).unsqueeze(2)
        ) & pg_mask.unsqueeze(2)

        phys_tok = req_to_token[
            reqs.view(n, 1, 1).expand(n, max_pages, self.page_size),
            tok_pos.clamp(0, req_to_token.shape[1] - 1),
        ].clamp(0, k_buffer.shape[0] - 1)

        keys = k_buffer[phys_tok].to(torch.float32)
        mask = tok_mask.unsqueeze(-1).unsqueeze(-1)

        # Mean pooling of keys within each page (page centroid representation).
        key_sum = torch.where(mask, keys, torch.zeros_like(keys)).sum(dim=2)
        tok_count = mask.sum(dim=2).clamp(min=1)
        page_mean = key_sum / tok_count

        phys_pg = (
            req_to_token[
                reqs.unsqueeze(1).expand(n, max_pages),
                tok_start.clamp(0, req_to_token.shape[1] - 1),
            ]
            // self.page_size
        )

        idx = pg_mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return

        target_pages = phys_pg[idx[:, 0], idx[:, 1]].clamp(
            0, self.page_key_mean[layer_id].shape[0] - 1
        )
        self.page_key_mean[layer_id][target_pages] = page_mean[idx[:, 0], idx[:, 1]]
        self.page_valid[layer_id][target_pages] = True

    def _align_query_heads(self, queries: torch.Tensor, head_dim: int, kv_heads: int):
        """Reshape query heads to align with stored KV heads (MQA/GQA-aware)."""
        if queries.dim() == 2:
            bs, hidden = queries.shape
            if hidden % head_dim != 0:
                raise ValueError(
                    f"Query hidden size {hidden} not divisible by head_dim {head_dim}"
                )
            q = queries.view(bs, hidden // head_dim, head_dim)
        elif queries.dim() == 3:
            q = queries
        else:
            raise ValueError(f"Unsupported query shape: {queries.shape}")

        q_heads = q.shape[1]
        if q_heads != kv_heads:
            if q_heads % kv_heads != 0:
                raise ValueError(
                    f"Query heads {q_heads} not divisible by KV heads {kv_heads}"
                )
            group = q_heads // kv_heads
            q = q.view(q.shape[0], kv_heads, group, head_dim).mean(dim=2)
        return q

    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        phys_pages_clamped = phys_pages.clamp(
            0, self.page_key_mean[layer_id].shape[0] - 1
        )

        means = self.page_key_mean[layer_id][phys_pages_clamped]
        valid_mask = self.page_valid[layer_id][phys_pages_clamped]

        head_dim = means.shape[-1]
        kv_heads = means.shape[-2]
        q = self._align_query_heads(queries, head_dim, kv_heads)
        q = q.to(means.dtype).unsqueeze(1)  # [bs, 1, kv_heads, head_dim]

        # Per-step attention proxy: query x page-centroid affinity, head-summed.
        affinity = (means * q).sum(dim=(2, 3))  # [bs, num_pages]
        affinity = torch.where(
            valid_mask, affinity, torch.full_like(affinity, float("-inf"))
        )

        # Softmax over pages turns the affinity into this step's attention share,
        # then fold it into the decayed running accumulator (ZSMerge core signal).
        step_attn = torch.softmax(affinity, dim=-1)
        prev = self.page_attn_accum[layer_id][phys_pages_clamped]
        accum = self.accum_decay * prev + step_attn
        self.page_attn_accum[layer_id][phys_pages_clamped] = accum

        scores = torch.where(valid_mask, accum, torch.full_like(accum, float("-inf")))
        return scores
