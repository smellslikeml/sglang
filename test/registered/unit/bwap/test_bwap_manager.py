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

import types
import unittest

import torch
from torch import nn

from sglang.srt.bwap.bwap_manager import (
    BWAPManager,
    _StepMode,
    build_topk_mask,
    compute_decode_scores,
    compute_prompt_scores,
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


def _make_manager(**overrides):
    kwargs = dict(sparsity=0.5, t_init=2, t_explore=1, t_prune=2)
    kwargs.update(overrides)
    return BWAPManager(base_model=TinyModel(), **kwargs)


def _forward_batch(forward_mode):
    # prepare_bwap_batch only reads forward_mode off the batch.
    return types.SimpleNamespace(forward_mode=forward_mode)


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
    def test_three_phase_schedule(self):
        manager = _make_manager()  # t_init=2, t_p=2, t_e=1, t_trans=3
        modes = []
        for _ in range(2 + 3 * 2):
            manager.prepare_bwap_batch(_forward_batch(ForwardMode.DECODE))
            modes.append(manager._mode)
        # Phase 2: T_init=2 explore steps. Then cycles of [P, P, E].
        self.assertEqual(
            modes,
            [
                _StepMode.EXPLORE,
                _StepMode.EXPLORE,
                _StepMode.PRUNE,
                _StepMode.PRUNE,
                _StepMode.EXPLORE,
                _StepMode.PRUNE,
                _StepMode.PRUNE,
                _StepMode.EXPLORE,
            ],
        )

    def test_extend_is_prompt_phase_and_does_not_advance_decode_step(self):
        manager = _make_manager()

        manager.prepare_bwap_batch(_forward_batch(ForwardMode.EXTEND))
        self.assertIs(manager._mode, _StepMode.PROMPT)
        self.assertEqual(manager.decode_step, 0)

    def test_realized_sparsity_discounts_dense_steps(self):
        manager = _make_manager()  # sparsity 0.5; per cycle: 2 prune, 1 explore
        for _ in range(2 + 3):
            manager.prepare_bwap_batch(_forward_batch(ForwardMode.DECODE))
        # dense = t_init(2) + explore(1) = 3, prune = 2
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
        manager = _make_manager(sparsity=0.0)
        manager.prepare_bwap_batch(_forward_batch(ForwardMode.EXTEND))
        out = _run_hooked(model, manager, x)
        for _ in range(3):  # past t_init, into a prune step
            manager.prepare_bwap_batch(_forward_batch(ForwardMode.DECODE))
        out = _run_hooked(model, manager, x)
        torch.testing.assert_close(out, dense)

    def test_pruned_step_applies_mask_and_stays_finite(self):
        model = TinyModel()
        x = torch.randn(4, HIDDEN)
        manager = _make_manager()

        # Prompt (s0) + t_init explore steps.
        manager.prepare_bwap_batch(_forward_batch(ForwardMode.EXTEND))
        _run_hooked(model, manager, x)
        for _ in range(2):
            manager.prepare_bwap_batch(_forward_batch(ForwardMode.DECODE))
            _run_hooked(model, manager, x)

        # First prune step: builds the mask from the max-aggregated scores
        # and applies it.
        manager.prepare_bwap_batch(_forward_batch(ForwardMode.DECODE))
        pruned = _run_hooked(model, manager, x)
        mask = manager.masks["mlp.act_fn"]
        self.assertEqual(int(mask.sum().item()), INTERMEDIATE // 2)
        self.assertTrue(bool(torch.isfinite(pruned).all()))

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
