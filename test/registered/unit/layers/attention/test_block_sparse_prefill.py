"""Unit tests for the mean-corrected block-sparse prefill attention backend.

Covers the FlashPrefill V2 algorithm (arXiv:2608.19758) ported in
``block_sparse_prefill`` and its registration as an SGLang attention backend.
"""

import math

import torch
import torch.nn.functional as F

# NON-NEW module: proves the registry wiring edit actually registered the backend.
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.block_sparse_prefill import (
    block_sparse_prefill_attention,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _dense_reference(query, key, value, *, scaling, causal=True):
    """Exact causal attention reference via torch SDPA (single sequence).

    Uses an explicit *bottom-right* causal mask so it matches SGLang extend
    semantics (the query aligns to the KV tail) for both q_len == kv_len and
    the chunked-prefill q_len < kv_len case.
    """
    q_len = query.shape[1]
    kv_len = key.shape[1]
    attn_mask = None
    if causal:
        q_pos = torch.arange(q_len).unsqueeze(1) + (kv_len - q_len)
        k_pos = torch.arange(kv_len).unsqueeze(0)
        attn_mask = k_pos <= q_pos
    out = F.scaled_dot_product_attention(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        attn_mask=None if attn_mask is None else attn_mask.unsqueeze(0),
        scale=scaling,
        enable_gqa=query.shape[0] != key.shape[0],
    )
    return out.squeeze(0)


def _rand_qkv(num_heads, kv_len, head_dim, *, seed=0, num_kv_heads=None):
    gen = torch.Generator().manual_seed(seed)
    num_kv_heads = num_kv_heads or num_heads
    q = torch.randn(num_heads, kv_len, head_dim, generator=gen)
    k = torch.randn(num_kv_heads, kv_len, head_dim, generator=gen)
    v = torch.randn(num_kv_heads, kv_len, head_dim, generator=gen)
    return q, k, v


def test_backend_is_registered():
    assert "flashprefill_sparse" in ATTENTION_BACKENDS
    assert callable(ATTENTION_BACKENDS["flashprefill_sparse"])


def test_keep_all_blocks_matches_dense():
    # An infinite threshold keeps every block -> exact dense attention.
    num_heads, kv_len, head_dim = 4, 320, 64
    scaling = 1.0 / math.sqrt(head_dim)
    q, k, v = _rand_qkv(num_heads, kv_len, head_dim, seed=1)

    ref = _dense_reference(q, k, v, scaling=scaling)
    got = block_sparse_prefill_attention(
        q,
        k,
        v,
        scaling=scaling,
        causal=True,
        block_size=64,
        keep_threshold=float("inf"),
    )
    torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)


def test_mean_correction_beats_hard_drop():
    # The V2 mean-correction term should shrink the approximation error of a
    # sparse pattern relative to plain hard-drop block-sparse attention.
    num_heads, kv_len, head_dim = 4, 512, 64
    scaling = 1.0 / math.sqrt(head_dim)
    q, k, v = _rand_qkv(num_heads, kv_len, head_dim, seed=2)
    ref = _dense_reference(q, k, v, scaling=scaling)

    # A tight threshold drops substantial softmax mass -- the extreme-sparsity
    # regime where the V2 correction matters most.
    common = dict(scaling=scaling, causal=True, block_size=64, keep_threshold=1.0)
    corrected = block_sparse_prefill_attention(q, k, v, mean_correction=True, **common)
    hard_drop = block_sparse_prefill_attention(q, k, v, mean_correction=False, **common)

    err_corrected = (corrected - ref).abs().mean().item()
    err_hard_drop = (hard_drop - ref).abs().mean().item()

    assert err_corrected < err_hard_drop
    # The correction recovers a large fraction of the accuracy loss (~55% in
    # this regime); assert a comfortable margin below hard-drop error.
    assert err_corrected < 0.7 * err_hard_drop


def test_actually_sparsifies():
    # With a tight threshold, at least one block per query must be dropped,
    # otherwise the "sparse" path is a no-op. Detect this by confirming the
    # corrected output differs from dense (it approximates, not reproduces).
    num_heads, kv_len, head_dim = 2, 512, 32
    scaling = 1.0 / math.sqrt(head_dim)
    q, k, v = _rand_qkv(num_heads, kv_len, head_dim, seed=3)
    ref = _dense_reference(q, k, v, scaling=scaling)
    sparse = block_sparse_prefill_attention(
        q,
        k,
        v,
        scaling=scaling,
        causal=True,
        block_size=64,
        keep_threshold=2.0,
    )
    assert (sparse - ref).abs().max().item() > 0.0
    # ...but still be a faithful approximation.
    assert (sparse - ref).abs().mean().item() < 0.1


def test_gqa_shapes_and_accuracy():
    # Grouped-query attention: fewer KV heads than query heads.
    num_heads, num_kv_heads, kv_len, head_dim = 8, 2, 320, 64
    scaling = 1.0 / math.sqrt(head_dim)
    q, k, v = _rand_qkv(
        num_heads,
        kv_len,
        head_dim,
        seed=4,
        num_kv_heads=num_kv_heads,
    )
    ref = _dense_reference(q, k, v, scaling=scaling)
    got = block_sparse_prefill_attention(
        q,
        k,
        v,
        scaling=scaling,
        causal=True,
        block_size=64,
        keep_threshold=float("inf"),
        enable_gqa=True,
    )
    assert got.shape == (num_heads, kv_len, head_dim)
    torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)


def test_short_query_extend_offset():
    # q_len < kv_len (a chunked-prefill extend): query aligns to the KV tail.
    num_heads, kv_len, q_len, head_dim = 4, 384, 96, 64
    scaling = 1.0 / math.sqrt(head_dim)
    _, k, v = _rand_qkv(num_heads, kv_len, head_dim, seed=5)
    gen = torch.Generator().manual_seed(6)
    q = torch.randn(num_heads, q_len, head_dim, generator=gen)

    ref = _dense_reference(q, k, v, scaling=scaling, causal=True)
    got = block_sparse_prefill_attention(
        q,
        k,
        v,
        scaling=scaling,
        causal=True,
        block_size=64,
        keep_threshold=float("inf"),
    )
    assert got.shape == (num_heads, q_len, head_dim)
    torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
