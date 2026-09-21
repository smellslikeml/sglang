"""
Hierarchical Sparse Attention (HSA) page selection.

Adapted from "Random Long-Context Access for Mamba via Hardware-aligned
Hierarchical Sparse Attention" (https://arxiv.org/abs/2504.16795). HSA gives an
RNN/Mamba backbone random access to long history by (1) splitting the KV cache
into fixed blocks, (2) compressing each block into a compact key representation,
and (3) scoring every block against the current query with a dot-product to
select the top-k relevant blocks before attending only within them.

This is an *adapted port* (Mode 2): the paper's core query-aware, block-level
top-k selection is kept at full fidelity, but the paper's *learned* block
compressor is replaced with a parameter-free landmark proxy -- the masked mean
of the block's keys. Mean pooling is the same block summary the base algorithm
docstring already anticipates for page reps, and it plugs directly into the
existing `_retrieve_page_scores`/`retrieve_topk` machinery. Unlike Quest's
min/max bounding-box *upper bound*, HSA scores a block with the true (scaled)
dot-product between the query and the compressed block key -- an
attention-aligned relevance estimate. The paper's softmax aggregation across
selected blocks is left to the attention backend and is intentionally out of
scope here.
"""

import logging
import math

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)

logger = logging.getLogger(__name__)


class HierarchicalSparseAlgorithm(BaseSparseAlgorithmImpl):
    """HSA page-wise sparse attention using compressed block landmarks.

    Each page is compressed to a single landmark key (masked mean of its keys);
    block relevance is the scaled query-landmark dot-product. Reuses the base
    class construct/update/retrieve flow -- only the representation and scoring
    steps are specialized.
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
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
            "Initialized HSA page reps: %d pages, %d layers, head_num=%d, head_dim=%d",
            total_num_pages,
            end_layer - start_layer,
            head_num,
            head_dim,
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

        # Masked mean over the page_size axis -> one landmark key per page.
        masked_keys = torch.where(mask, keys, torch.zeros_like(keys))
        counts = tok_mask.sum(dim=2).clamp(min=1).unsqueeze(-1).unsqueeze(-1)
        page_mean = masked_keys.sum(dim=2) / counts

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
        # Clamp pages to valid storage range.
        phys_pages_clamped = phys_pages.clamp(
            0, self.page_k_mean[layer_id].shape[0] - 1
        )

        k_mean = self.page_k_mean[layer_id][phys_pages_clamped]
        valid_mask = self.page_valid[layer_id][phys_pages_clamped]

        # Align query shape to KV heads.
        head_dim = k_mean.shape[-1]
        if queries.dim() == 2:
            bs, hidden = queries.shape
            if hidden % head_dim != 0:
                raise ValueError(
                    f"HSA query hidden size {hidden} not divisible by head_dim {head_dim}"
                )
            q_heads = hidden // head_dim
            q = queries.view(bs, q_heads, head_dim)
        elif queries.dim() == 3:
            q = queries
        else:
            raise ValueError(f"Unsupported query shape for HSA: {queries.shape}")

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

        # Attention-aligned block relevance: scaled query-landmark dot product.
        scale = 1.0 / math.sqrt(head_dim)
        relevance = (q * k_mean).sum(dim=(2, 3)) * scale
        relevance = torch.where(
            valid_mask, relevance, torch.full_like(relevance, float("-inf"))
        )

        return relevance
