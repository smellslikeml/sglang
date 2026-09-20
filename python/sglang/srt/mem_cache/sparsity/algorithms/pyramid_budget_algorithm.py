"""Layer-wise pyramid KV-cache budget with attention-aligned page importance.

Adapted from PyramidInfer (https://arxiv.org/abs/2405.12532). PyramidInfer's
central observation is that deeper transformer layers concentrate their
attention mass on progressively fewer "pivotal" context tokens, so a *uniform*
per-layer retention budget over-provisions the deep layers. Its two
contributions map onto this framework as:

1. **Pivotal-context importance scoring** -> ``_compute_page_representations`` /
   ``_retrieve_page_scores``. PyramidInfer ranks past tokens by the attention
   they receive from a recent observation window. This retain-full-cache
   framework does not record prefill attention weights, so (Mode-2 adaptation,
   the same retain-semantics substitution ``base_algorithm.py`` documents for
   SnapKV) we use a parameter-free proxy: a mean-pooled key per page scored
   against the current query, ``q . mean(k)``, which tracks the attention mass a
   page would receive without materializing the full score matrix.

2. **Decreasing per-layer "pyramid" budget** -> a layer-dependent override of
   ``sparsity_ratio`` via ``_sparsity_ratio_for_layer``. This is the paper's
   flagship mechanism and is kept at full fidelity: the retained fraction decays
   monotonically from the shallowest to the deepest sparse layer.

Intentionally out of scope for this integration (the paper's method, not its
result): recording real prefill attention weights, the decode-time PvC growth
schedule, and PyramidInfer's separate eviction pathway -- all of which belong to
the eviction-origin formulation this framework deliberately reframes as
query-aware top-k retrieval.
"""

import logging

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)

logger = logging.getLogger(__name__)


class PyramidBudgetAlgorithm(BaseSparseAlgorithmImpl):
    """Page-wise sparse attention with a layer-decaying (pyramid) retention budget.

    Extra config keys (all under ``sparse_extra_config``):
        pyramid_max_ratio: retained fraction at the shallowest sparse layer.
            Defaults to ``sparsity_ratio``.
        pyramid_min_ratio: retained fraction at the deepest sparse layer.
            Defaults to ``0.5 * pyramid_max_ratio``.
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        extra = config.sparse_extra_config
        self.pyramid_max_ratio = float(
            extra.get("pyramid_max_ratio", self.sparsity_ratio)
        )
        self.pyramid_min_ratio = float(
            extra.get("pyramid_min_ratio", 0.5 * self.pyramid_max_ratio)
        )
        if not 0.0 < self.pyramid_min_ratio <= self.pyramid_max_ratio <= 1.0:
            raise ValueError(
                "pyramid ratios must satisfy 0 < min <= max <= 1, got "
                f"min={self.pyramid_min_ratio}, max={self.pyramid_max_ratio}"
            )
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
            "Initialized PyramidBudget page reps: %d pages, layers [%d,%d), "
            "ratio %.3f->%.3f, head_num=%d, head_dim=%d",
            total_num_pages,
            start_layer,
            end_layer,
            self.pyramid_max_ratio,
            self.pyramid_min_ratio,
            head_num,
            head_dim,
        )

    def _sparsity_ratio_for_layer(self, layer_id: int) -> float:
        """Pyramid budget: retained fraction decays linearly with layer depth."""
        span = max(self.end_layer - 1 - self.start_layer, 1)
        depth = min(max(layer_id - self.start_layer, 0), span)
        frac = depth / span
        return self.pyramid_max_ratio - frac * (
            self.pyramid_max_ratio - self.pyramid_min_ratio
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

        masked_keys = torch.where(mask, keys, torch.zeros_like(keys))
        token_counts = tok_mask.sum(dim=2).clamp(min=1)  # [n, max_pages]
        page_mean = masked_keys.sum(dim=2) / token_counts.unsqueeze(-1).unsqueeze(-1)

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
                    f"PyramidBudget query hidden size {hidden} not divisible by "
                    f"head_dim {head_dim}"
                )
            q = queries.view(bs, hidden // head_dim, head_dim)
        elif queries.dim() == 3:
            q = queries
        else:
            raise ValueError(
                f"Unsupported query shape for PyramidBudget: {queries.shape}"
            )

        kv_heads = k_mean.shape[-2]
        q_heads = q.shape[1]
        if q_heads != kv_heads:
            if q_heads % kv_heads != 0:
                raise ValueError(
                    f"Query heads {q_heads} not divisible by KV heads {kv_heads}"
                )
            group = q_heads // kv_heads
            # Average grouped query heads to align with KV heads (MQA/GQA proxy).
            q = q.view(q.shape[0], kv_heads, group, head_dim).mean(dim=2)

        q = q.to(k_mean.dtype).unsqueeze(1)  # [bs, 1, kv_heads, head_dim]

        # Query-page attention proxy: q . mean(k), summed over heads and dims.
        scores = (q * k_mean).sum(dim=(2, 3))
        scores = torch.where(valid_mask, scores, torch.full_like(scores, float("-inf")))
        return scores
