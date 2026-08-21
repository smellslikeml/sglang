# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Unit tests for the Phase-1 functional BWAP integration."""

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.bwap.bwap_fused import (
    fast_path_eligible,
    fused_pruned_mlp,
    gather_ffn_weights,
    silu_and_mul,
)
from sglang.srt.bwap.bwap_manager import (
    BWAPManager,
    build_topk_mask,
    compute_decode_scores,
    compute_prompt_scores,
    compute_row_modes,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 16
INTERMEDIATE = 32


class TinyGatedMLP(nn.Module):
    """Qwen2MLP-shaped module: the hook target is ``mlp.act_fn``."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(HIDDEN, 2 * INTERMEDIATE, bias=False)
        self.act_fn = SiluAndMul()
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.mlp = TinyGatedMLP()

    def forward(self, x):
        return self.mlp(x)


def _make_manager(model=None, **overrides):
    # Bind the manager to the model under test so register_hooks() attaches to
    # the very modules the test forwards run through. Schedule-only tests, which
    # never forward a model, may omit it and get a throwaway one.
    kwargs = dict(sparsity=0.5, t_init=2, t_explore=1, t_prune=2)
    kwargs.update(overrides)
    return BWAPManager(base_model=model if model is not None else TinyModel(), **kwargs)


def _extend(manager, *, req_ids, prompt_len=5):
    manager.prepare_bwap_batch(
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.tensor(req_ids),
        seq_lens=torch.tensor([prompt_len] * len(req_ids)),
    )


def _decode(manager, *, req_ids, seq_lens):
    manager.prepare_bwap_batch(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.tensor(req_ids),
        seq_lens=torch.tensor(seq_lens),
    )


def _run_hooked(model, manager, x):
    manager.register_hooks()
    try:
        with torch.no_grad():
            return model(x)
    finally:
        manager.remove_hooks()


class TestBWAPScores(CustomTestCase):
    def test_prompt_scores_eq2(self):
        z = torch.tensor([[3.0, 4.0, 0.0], [0.0, 1.0, 0.0]])
        # Row-normalized: [[0.6, 0.8, 0], [0, 1, 0]]; column-L2 / sqrt(2).
        expected = torch.tensor(
            [
                (0.6**2 + 0.0**2) ** 0.5,
                (0.8**2 + 1.0**2) ** 0.5,
                0.0,
            ]
        ) / (2**0.5)
        torch.testing.assert_close(compute_prompt_scores(z), expected)

    def test_decode_scores_are_batch_max_of_normalized_rows(self):
        # Two active sequences with disjoint large neurons: the shared
        # batch-aggregated score (Eq. 3, element-wise max) keeps both.
        z = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        scores = compute_decode_scores(z)
        torch.testing.assert_close(scores, torch.ones(2))

    def test_topk_mask_keeps_round_1_minus_sparsity_of_dim(self):
        scores = torch.arange(10, dtype=torch.float32)
        mask = build_topk_mask(scores, sparsity=0.5)
        self.assertEqual(int(mask.sum().item()), 5)
        # The kept neurons are the highest-scoring ones.
        self.assertTrue(bool((mask[5:] == 1.0).all()))
        self.assertTrue(bool((mask[:5] == 0.0).all()))


class TestBWAPSchedule(CustomTestCase):
    def test_single_request_three_phase_schedule(self):
        # Derived property: one request's decode steps 0..7 follow T_init(2)
        # explore steps, then cycles of [prune, prune, explore] (t_p=2, t_e=1).
        manager = _make_manager()
        _extend(manager, req_ids=[0])
        pruned = []
        for step in range(8):
            _decode(manager, req_ids=[0], seq_lens=[5 + step])
            pruned.append(bool(manager._prune_rows[0]))
        self.assertEqual(pruned, [False, False, True, True, False, True, True, False])

    def test_late_joiner_gets_its_own_t_init(self):
        # Bug regression: with a single global step counter, a request joining a
        # running batch mid-cycle is pruned immediately using a mask built from
        # the other requests' activations, never getting its own exploration.
        # Per-request scheduling must let the late joiner explore first.
        manager = _make_manager()
        _extend(manager, req_ids=[0])
        for step in range(6):  # push req 0 well past T_init into a prune phase
            _decode(manager, req_ids=[0], seq_lens=[5 + step])
        self.assertTrue(bool(manager._prune_rows[0]))

        _extend(manager, req_ids=[1])  # a new request joins
        _decode(manager, req_ids=[0, 1], seq_lens=[11, 5])  # mixed decode
        self.assertTrue(bool(manager._prune_rows[0]))  # established req still prunes
        self.assertFalse(bool(manager._prune_rows[1]))  # late joiner explores
        self.assertTrue(bool(manager._collect_rows[1]))

    def test_synchronized_batch_reduces_to_global_schedule(self):
        # Derived property: when every row shares a step, the per-row modes are
        # the paper's single global schedule — the whole batch prunes together
        # or explores together, and the two masks are exact complements.
        manager = _make_manager()
        for step in range(12):
            steps = torch.full((4,), step)
            prune, collect = compute_row_modes(
                steps,
                t_init=manager.t_init,
                t_prune=manager.t_prune,
                t_trans=manager.t_trans,
            )
            self.assertTrue(bool(prune.all()) or bool(collect.all()))
            self.assertTrue(torch.equal(prune, ~collect))

    def test_realized_sparsity_discounts_dense_steps(self):
        # sparsity 0.5; decode steps 0..4 are E,E,P,P,E -> 2 pruned of 5.
        manager = _make_manager()
        _extend(manager, req_ids=[0])
        for step in range(5):
            _decode(manager, req_ids=[0], seq_lens=[5 + step])
        self.assertAlmostEqual(manager.realized_sparsity, 0.5 * 2 / 5)


class TestBWAPHookIntegration(CustomTestCase):
    def test_hook_targets_found_on_mlp_act_fn(self):
        manager = _make_manager()
        self.assertEqual(list(manager.hook_targets.keys()), ["mlp.act_fn"])

    def test_all_ones_mask_reproduces_dense_output(self):
        model = TinyModel()
        x = torch.randn(4, HIDDEN)
        with torch.no_grad():
            dense = model(x)

        # sparsity=0 -> k = D -> all-ones mask -> dense output exactly.
        manager = _make_manager(model, sparsity=0.0)
        _extend(manager, req_ids=[0, 1, 2, 3])
        _run_hooked(model, manager, x)
        for step in range(3):  # past t_init, into a prune step
            _decode(manager, req_ids=[0, 1, 2, 3], seq_lens=[5 + step] * 4)
        out = _run_hooked(model, manager, x)
        torch.testing.assert_close(out, dense)

    def test_pruned_step_applies_mask_and_stays_finite(self):
        model = TinyModel()
        x = torch.randn(4, HIDDEN)
        manager = _make_manager(model)

        _extend(manager, req_ids=[0, 1, 2, 3])
        _run_hooked(model, manager, x)  # collect s0
        for step in range(2):  # t_init explore steps
            _decode(manager, req_ids=[0, 1, 2, 3], seq_lens=[5 + step] * 4)
            _run_hooked(model, manager, x)

        # First prune step: build the mask from the max-aggregated scores.
        _decode(manager, req_ids=[0, 1, 2, 3], seq_lens=[7] * 4)
        pruned = _run_hooked(model, manager, x)
        mask = manager.masks["mlp.act_fn"]
        self.assertEqual(int(mask.sum().item()), INTERMEDIATE // 2)
        self.assertTrue(bool(torch.isfinite(pruned).all()))

    def test_mixed_batch_prunes_only_in_cycle_rows(self):
        # Bug regression (end-to-end): in one decode forward with an established
        # request (pruning) and a late joiner (exploring), the returned
        # activation must mask only the established row and leave the joiner
        # dense — a single global mask would wrongly prune both.
        model = TinyModel()
        manager = _make_manager(model)
        _extend(manager, req_ids=[0])
        _run_hooked(model, manager, torch.randn(3, HIDDEN))  # prompt s0 -> mem
        for step in range(6):  # real explore/prune forwards populate mem + mask
            _decode(manager, req_ids=[0], seq_lens=[5 + step])
            _run_hooked(model, manager, torch.randn(1, HIDDEN))
        _extend(manager, req_ids=[1])
        _decode(manager, req_ids=[0, 1], seq_lens=[11, 5])  # req0 prunes, req1 explores

        z = torch.randn(2, INTERMEDIATE)
        out = manager._process_activation("mlp.act_fn", z)
        mask = manager.masks["mlp.act_fn"].bool()
        self.assertEqual(
            int(mask.sum().item()), INTERMEDIATE // 2
        )  # a real top-k, not all-ones
        torch.testing.assert_close(out[0], z[0] * mask)  # established: pruned
        torch.testing.assert_close(out[1], z[1])  # late joiner: dense

    def test_mask_refresh_follows_mem_version(self):
        manager = _make_manager()
        z = torch.randn(4, INTERMEDIATE)
        scores = torch.arange(INTERMEDIATE, dtype=torch.float32)
        manager._update_mem("mlp.act_fn", scores)
        mask1 = manager._get_mask("mlp.act_fn", z)
        # A new max-aggregated score invalidates the cached mask; scaling the
        # flipped scores guarantees the running max (and its top-k) changes.
        manager._update_mem("mlp.act_fn", scores.flip(0) * 2)
        mask2 = manager._get_mask("mlp.act_fn", z)
        self.assertFalse(torch.equal(mask1, mask2))


class TestBWAPFused(CustomTestCase):
    def test_gather_ffn_parity(self):
        # Derived property: the gather-GEMM equals the masked dense FFN. Uses
        # realistic (1/sqrt(fan_in)) init so activations are O(1) and fp reorder
        # is the only difference.
        torch.manual_seed(0)
        hid, d_ff, k = 32, 96, 48
        gate_up_w = torch.randn(2 * d_ff, hid) / hid**0.5
        down_w = torch.randn(hid, d_ff) / d_ff**0.5
        x = torch.randn(4, hid)
        keep = torch.topk(torch.rand(d_ff), k).indices.sort().values
        mask = torch.zeros(d_ff)
        mask[keep] = 1.0

        y_dense = F.linear(silu_and_mul(F.linear(x, gate_up_w)) * mask, down_w)
        gate_up_k, down_k = gather_ffn_weights(gate_up_w, down_w, keep, d_ff)
        y_fused = fused_pruned_mlp(x, gate_up_k, down_k)
        torch.testing.assert_close(y_dense, y_fused, rtol=1e-5, atol=1e-6)

    def test_eligibility_guards(self):
        model = TinyModel()
        gu, dn = model.mlp.gate_up_proj, model.mlp.down_proj
        self.assertTrue(fast_path_eligible(gu, dn, tp_size=1))  # bias-free float, TP=1
        self.assertFalse(fast_path_eligible(gu, dn, tp_size=2))  # TP>1 needs a reduce
        biased = nn.Linear(HIDDEN, INTERMEDIATE, bias=True)
        self.assertFalse(fast_path_eligible(biased, dn, tp_size=1))  # bias unsupported

    def test_fused_forward_equals_masked_dense(self):
        # Bug regression + derived property: with --bwap-fused, an all-prune
        # decode step must return the SAME output as the Phase-1 masked dense
        # path (the fused path is a pure-perf refactor, not a behavior change).
        model = TinyModel()
        x = torch.randn(4, HIDDEN)
        mgr = BWAPManager(
            base_model=model,
            sparsity=0.5,
            t_init=2,
            t_explore=1,
            t_prune=2,
            fused=True,
            tp_size=1,
        )
        mgr.register_hooks()
        mgr.install_fused_forwards()
        try:
            self.assertTrue(all(mgr._fast_ok.values()))  # TinyGatedMLP is eligible
            _extend(mgr, req_ids=[0, 1, 2, 3])
            with torch.no_grad():
                model(x)  # prompt: collect s0
            for step in range(2):  # T_init explore steps: collect, dense forward
                _decode(mgr, req_ids=[0, 1, 2, 3], seq_lens=[5 + step] * 4)
                with torch.no_grad():
                    model(x)
            # step 2 -> all-prune: the wrapper takes the gather-GEMM fast path
            _decode(mgr, req_ids=[0, 1, 2, 3], seq_lens=[7] * 4)
            self.assertTrue(mgr._all_prune)
            with torch.no_grad():
                y_fused = model(x)
        finally:
            mgr.remove_fused_forwards()
            mgr.remove_hooks()

        key = next(iter(mgr.gated_mlps))
        mask = mgr.masks[key]
        self.assertEqual(int(mask.sum().item()), INTERMEDIATE // 2)  # real top-k
        gu_w, dn_w = model.mlp.gate_up_proj.weight, model.mlp.down_proj.weight
        y_ref = F.linear(silu_and_mul(F.linear(x, gu_w)) * mask, dn_w)  # masked dense
        torch.testing.assert_close(y_fused, y_ref, rtol=1e-4, atol=1e-5)

    def test_graph_gating_opens_after_warmup_freeze(self):
        # Phase 2b: armed gating keeps graph_ready() False until `warmup` decode
        # forwards have built the mask; then _freeze_and_fill fills the buffers and
        # opens the gate. Guards the eager-warmup → frozen-graph handoff.
        model = TinyModel()
        mgr = BWAPManager(
            base_model=model,
            sparsity=0.5,
            t_init=2,
            t_explore=1,
            t_prune=2,
            fused=True,
            tp_size=1,
        )
        mgr.install_fused_forwards()
        mgr.begin_capture()  # allocate fixed dummy buffers (as at capture)
        mgr.end_capture()
        mgr.arm_graph_gating()
        key = next(iter(mgr.gated_mlps))
        self.assertFalse(mgr.graph_ready())  # gated until warmup

        mgr._update_mem(key, torch.rand(INTERMEDIATE))  # warmup scoring
        _extend(mgr, req_ids=[0])
        for step in range(2):  # _warmup_steps == t_init == 2 decode forwards
            self.assertFalse(mgr.graph_ready())  # still eager mid-warmup
            _decode(mgr, req_ids=[0], seq_lens=[5 + step])
        self.assertTrue(mgr.graph_ready())  # gate opened
        self.assertIn(key, mgr._gate_up_buf)  # buffers filled from the real mask
        self.assertEqual(int(mgr.masks[key].sum().item()), INTERMEDIATE // 2)

    def test_gathered_buffers_keep_stable_address_across_refresh(self):
        # Capture-readiness (Phase 2b): the gathered-weight buffers must refresh
        # in place (copy_), never be reassigned — a captured CUDA graph replays
        # against a fixed pointer. Guards a regression back to per-refresh alloc.
        model = TinyModel()
        mgr = BWAPManager(
            base_model=model,
            sparsity=0.5,
            t_init=2,
            t_explore=1,
            t_prune=2,
            fused=True,
            tp_size=1,
        )
        key = next(iter(mgr.gated_mlps))
        mlp = mgr.gated_mlps[key]
        x = torch.randn(4, HIDDEN)

        mgr._update_mem(key, torch.rand(INTERMEDIATE))  # first mask
        mgr._fast_mlp_forward(key, mlp, x)  # allocates the fixed buffers
        buf_gu, buf_dn = mgr._gate_up_buf[key], mgr._down_buf[key]
        ptr_gu, ptr_dn = buf_gu.data_ptr(), buf_dn.data_ptr()

        mgr._update_mem(key, torch.rand(INTERMEDIATE) * 5)  # different mask -> refresh
        mgr._fast_mlp_forward(key, mlp, x)
        self.assertIs(mgr._gate_up_buf[key], buf_gu)  # same tensor object
        self.assertIs(mgr._down_buf[key], buf_dn)
        self.assertEqual(mgr._gate_up_buf[key].data_ptr(), ptr_gu)  # same address
        self.assertEqual(mgr._down_buf[key].data_ptr(), ptr_dn)


if __name__ == "__main__":
    unittest.main()
