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
compressor is replaced with a parameter-free landmark proxy. Unlike Quest's
min/max bounding-box *upper bound*, HSA scores a block with the true (scaled)
dot-product between the query and the compressed block landmark(s) -- an
attention-aligned relevance estimate. The paper's softmax aggregation across
selected blocks is left to the attention backend and is intentionally out of
scope here.

To capture the paper's core innovation -- *fine-grained token-level
information inside each chunk* -- each block is compressed into
``num_landmarks`` sub-segment landmarks (masked means of contiguous token
sub-segments) and its relevance is the maximum over those landmarks. This
keeps a chunk that holds a single strongly-matching token in the top-k even
when its other tokens are unrelated. With ``num_landmarks == 1`` this reduces
exactly to a single masked-mean landmark per block. See `landmark_pooling`.
"""

import logging
import math

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
    BaseSparseAlgorithmImpl,
)
from sglang.srt.mem_cache.sparsity.algorithms.landmark_pooling import (
    pool_page_landmarks,
    score_pages_by_landmarks,
)

logger = logging.getLogger(__name__)


class HierarchicalSparseAlgorithm(BaseSparseAlgorithmImpl):
    """HSA page-wise sparse attention using compressed block landmarks.

    Each page is compressed to ``num_landmarks`` sub-segment landmark keys
    (masked means of contiguous token sub-segments); block relevance is the
    maximum scaled query-landmark dot-product over those landmarks. Reuses the
    base class construct/update/retrieve flow -- only the representation and
    scoring steps are specialized. ``num_landmarks`` defaults to 1, in which
    case a page is a single masked-mean landmark.
    """

    def __init__(self, config, device: torch.device, **kwargs):
        super().__init__(config, device, **kwargs)
        self.num_landmarks = int(config.sparse_extra_config.get("num_landmarks", 1))
        if self.num_landmarks < 1:
            raise ValueError(f"num_landmarks must be >= 1, got {self.num_landmarks}")
        self.page_k_landmarks = {}
        self.page_valid = {}

    def _initialize_representation_pools(
        self, start_layer: int, end_layer: int, total_num_pages: int
    ):
        key_buf = self.token_to_kv_pool.get_key_buffer(start_layer)
        head_num, head_dim = key_buf.shape[1], key_buf.shape[2]

        for layer_id in range(start_layer, end_layer):
            self.page_k_landmarks[layer_id] = torch.zeros(
                (total_num_pages, self.num_landmarks, head_num, head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.page_valid[layer_id] = torch.zeros(
                total_num_pages, dtype=torch.bool, device=self.device
            )

        logger.info(
            "Initialized HSA page reps: %d pages, %d layers, "
            "num_landmarks=%d, head_num=%d, head_dim=%d",
            total_num_pages,
            end_layer - start_layer,
            self.num_landmarks,
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

        # Split each page into num_landmarks sub-segments, masked-mean each ->
        # [n, max_pages, num_landmarks, head_num, head_dim]. Fine-grained
        # sub-segment landmarks preserve within-page token structure the paper
        # relies on for precise chunk selection.
        page_landmarks = pool_page_landmarks(keys, tok_mask, self.num_landmarks)

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
            0, self.page_k_landmarks[layer_id].shape[0] - 1
        )
        self.page_k_landmarks[layer_id][target_pages] = page_landmarks[
            idx[:, 0], idx[:, 1]
        ]
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
            0, self.page_k_landmarks[layer_id].shape[0] - 1
        )

        # [bs, num_pages, num_landmarks, kv_heads, head_dim]
        landmarks = self.page_k_landmarks[layer_id][phys_pages_clamped]
        valid_mask = self.page_valid[layer_id][phys_pages_clamped]

        # Align query shape to KV heads.
        head_dim = landmarks.shape[-1]
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

        kv_heads = landmarks.shape[-2]
        q_heads = q.shape[1]
        if q_heads != kv_heads:
            if q_heads % kv_heads != 0:
                raise ValueError(
                    f"Query heads {q_heads} not divisible by KV heads {kv_heads}"
                )
            group = q_heads // kv_heads
            # Average grouped query heads to align with KV heads (MQA/GQA proxy).
            q = q.view(q.shape[0], kv_heads, group, head_dim).mean(dim=2)

        q = q.to(landmarks.dtype)  # [bs, kv_heads, head_dim]

        # Attention-aligned block relevance: max over sub-segment landmarks of
        # the scaled query-landmark dot product. The max keeps a block whose
        # single strongly-matching sub-segment would be diluted by a mean.
        scale = 1.0 / math.sqrt(head_dim)
        relevance = score_pages_by_landmarks(q, landmarks, scale)
        relevance = torch.where(
            valid_mask, relevance, torch.full_like(relevance, float("-inf"))
        )

        return relevance
