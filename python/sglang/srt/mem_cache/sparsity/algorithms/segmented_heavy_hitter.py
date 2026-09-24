"""
Segmented heavy-hitter sparse attention algorithm.

Instead of selecting the globally top-scoring KV pages (as Quest does), this
algorithm partitions the request history into fixed-size *segments* and keeps a
few heavy hitters within each segment, together with an always-retained block of
attention-sink pages at the start and a local window of recent pages at the end.
Retaining heavy hitters segment-locally keeps the selected KV spatially
distributed across the context rather than clustered, which preserves the
contextual structure that global top-k tends to collapse. Per-segment budgets
decay geometrically with distance from the local window (segments nearer the
recent context keep more pages), giving the "beehive" retention profile.

Adapted from:
    BUZZ: Beehive-structured Sparse KV Cache with Segmented Heavy Hitters for
    Efficient LLM Inference (https://arxiv.org/abs/2410.23079).

Adaptation notes (this is a target-native adapted port, not a direct port):
    - The paper identifies heavy hitters from *accumulated* attention
      probabilities gathered across every decode step. The sparse coordinator
      here does not expose an attention-probability accumulation hook, so we
      substitute a parameter-free query-key attention-mass proxy: mean-pooled
      per-page keys scored against the current decode query (the same retrieval
      contract Quest already uses). The segmented/beehive selection structure —
      the paper's actual contribution — is kept at full fidelity.
    - The paper's ring-buffer eviction / memory bookkeeping is intentionally not
      reimplemented here; page offloading is handled separately by the
      SparseCoordinator / backend adaptor.
"""

import logging

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)

logger = logging.getLogger(__name__)


class SegmentedHeavyHitterAlgorithm(BaseSparseAlgorithmImpl):
    """Page-wise sparse attention with segmented heavy hitters and sink pages."""

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        extra = config.sparse_extra_config
        # Number of history pages per segment (heavy hitters are picked within
        # each segment independently).
        self.segment_pages = max(int(extra.get("segment_pages", 8)), 1)
        # Leading attention-sink pages that are always retained.
        self.num_sink_pages = max(int(extra.get("num_sink_pages", 1)), 0)
        # Geometric decay applied to the per-segment budget as segments get
        # farther from the local window (1.0 == uniform budget per segment).
        self.beehive_decay = float(extra.get("beehive_decay", 0.9))
        self.page_k_mean = {}
        self.page_valid = {}

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)
        head_num, head_dim = key_buf.shape[1], key_buf.shape[2]

        for layer_id in range(start_layer, end_layer):
            self.page_k_mean[layer_id] = torch.zeros(
                (total_num_pages, head_num, head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.page_valid[layer_id] = torch.zeros(
                total_num_pages, dtype=torch.bool, device=self.device
            )

        logger.info(
            "Initialized segmented heavy-hitter reps: %d pages, %d layers, "
            "segment_pages=%d, num_sink_pages=%d, beehive_decay=%.3f",
            total_num_pages,
            end_layer - start_layer,
            self.segment_pages,
            self.num_sink_pages,
            self.beehive_decay,
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
        mask = tok_mask.unsqueeze(-1).unsqueeze(-1).to(keys.dtype)

        # Masked mean over the page_size axis -> per-page mean key ("heavy-hitter
        # representation"). Guard against empty pages to avoid division by zero.
        counts = mask.sum(dim=2).clamp(min=1.0)
        page_mean = (keys * mask).sum(dim=2) / counts

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
            0, self.page_k_mean[layer_id].shape[0] - 1
        )
        self.page_k_mean[layer_id][target_pages] = page_mean[idx[:, 0], idx[:, 1]]
        self.page_valid[layer_id][target_pages] = True

    def _retrieve_page_scores(
        self,
        layer_id: int,
        phys_pages: torch.Tensor,
        req_pool_indices: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        phys_pages_clamped = phys_pages.clamp(
            0, self.page_k_mean[layer_id].shape[0] - 1
        )

        k_mean = self.page_k_mean[layer_id][phys_pages_clamped]
        valid_mask = self.page_valid[layer_id][phys_pages_clamped]

        head_dim = k_mean.shape[-1]
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

        kv_heads = k_mean.shape[-2]
        q_heads = q.shape[1]
        if q_heads != kv_heads:
            if q_heads % kv_heads != 0:
                raise ValueError(
                    f"Query heads {q_heads} not divisible by KV heads {kv_heads}"
                )
            group = q_heads // kv_heads
            q = q.view(q.shape[0], kv_heads, group, head_dim).mean(dim=2)

        q = q.to(k_mean.dtype).unsqueeze(1)  # [bs, 1, kv_heads, head_dim]

        # Attention-mass proxy: dot(query, mean-pooled page key) summed over heads.
        scores = (q * k_mean).sum(dim=(2, 3))
        scores = torch.where(
            valid_mask, scores, torch.full_like(scores, float("-inf"))
        )
        return scores

    def _segment_budget(self, seg_rank: int, seg_len: int) -> int:
        """Heavy-hitter budget for a segment ``seg_rank`` steps from the local
        window (0 == nearest). Budget decays geometrically with distance and is
        clamped to ``[1, seg_len]``."""
        base = self.segment_pages * self.sparsity_ratio
        budget = int(round(base * (self.beehive_decay**seg_rank)))
        return max(1, min(budget, seg_len))

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """Segmented heavy-hitter retrieval: attention-sink pages + per-segment
        heavy hitters (with beehive-decaying budgets) + local window."""
        bs, device = queries.shape[0], queries.device

        seq_lens_source = kwargs.get("forward_batch", None)
        if seq_lens_source is None or not hasattr(seq_lens_source, "seq_lens"):
            raise ValueError(
                "forward_batch with seq_lens is required for TopK retrieval"
            )
        seq_lens = seq_lens_source.seq_lens.to(device)

        req_to_token = self.req_to_token_pool.req_to_token
        max_req_tokens = req_to_token.shape[1]

        per_request_indices = []
        per_request_lengths = []

        for i in range(bs):
            num_pages = int((seq_lens[i].item() + self.page_size - 1) // self.page_size)
            # No sparsification needed when everything already fits in the
            # always-retained sink + local window.
            if (
                not sparse_mask[i]
                or num_pages <= self.num_recent_pages + self.num_sink_pages
            ):
                per_request_indices.append(
                    torch.empty(0, device=device, dtype=torch.int32)
                )
                per_request_lengths.append(0)
                continue

            page_idx = torch.arange(num_pages, device=device)
            page_start_token = req_to_token[
                req_pool_indices[i],
                (page_idx * self.page_size).clamp(0, max_req_tokens - 1),
            ]
            phys_pages = (page_start_token // self.page_size).unsqueeze(0)

            scores = self._retrieve_page_scores(
                layer_id,
                phys_pages,
                req_pool_indices[i : i + 1],
                queries[i : i + 1],
            ).squeeze(0)

            recent_start = num_pages - self.num_recent_pages
            history_start = self.num_sink_pages
            history_end = recent_start

            selected = [
                torch.arange(0, self.num_sink_pages, device=device),
                torch.arange(recent_start, num_pages, device=device),
            ]

            # Walk segments from the local window backwards toward the sinks so
            # that seg_rank grows with distance (beehive budget decay).
            seg_rank = 0
            seg_hi = history_end
            while seg_hi > history_start:
                seg_lo = max(seg_hi - self.segment_pages, history_start)
                seg_len = seg_hi - seg_lo
                seg_scores = scores[seg_lo:seg_hi]
                budget = self._segment_budget(seg_rank, seg_len)
                topk_local = torch.topk(seg_scores, k=budget, sorted=False)[1]
                selected.append(topk_local + seg_lo)
                seg_hi = seg_lo
                seg_rank += 1

            combined = torch.unique(torch.cat(selected).to(torch.int32))
            per_request_indices.append(combined)
            per_request_lengths.append(int(combined.numel()))

        max_len = max(max(per_request_lengths, default=0), 1)
        out_indices = torch.full((bs, max_len), -1, dtype=torch.int32, device=device)
        out_lengths = torch.zeros(bs, dtype=torch.int32, device=device)

        for i, sel in enumerate(per_request_indices):
            length = per_request_lengths[i]
            if length == 0:
                continue
            out_indices[i, :length] = sel
            out_lengths[i] = length

        return out_indices, out_lengths
