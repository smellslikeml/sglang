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
from torch import nn

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


if __name__ == "__main__":
    unittest.main()
