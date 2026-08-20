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

"""Batch-wise Adaptive Pruning (BWAP), Phase 1: functional integration.

Training-free, inference-time FFN neuron pruning for gated-MLP LLMs, built
for batched decode (arXiv:2608.14003). The gated MLP's activation function
(``SiluAndMul``) produces ``Z`` of shape ``[tokens, intermediate]``, which
feeds ``down_proj``. BWAP scores the neurons of ``Z`` per layer, keeps the
top-``k`` (``k = round((1 - sparsity) * D_FF)``), and zeroes the rest via a
shared per-layer binary mask: ``Z * mask`` (functional; numerically
identical to a gather-based pruning, without fused kernels).

Phase schedule per decode step (defaults ``T_init=8, T_E=4, T_p=16``):

- Prompt (extend) forward: collect the prompt score ``s0``; stay dense.
- Phase 2, first ``T_init`` decode steps: dense exploration, updating the
  running max-aggregated score ``m``; the first mask is built afterwards.
- Phase 3, cycles of ``T_trans = T_E + T_p`` steps: ``T_p`` sparse steps
  (apply mask) followed by ``T_E`` dense-explore steps (refresh ``m``);
  the mask is rebuilt from the latest ``m`` at the next sparse step.

Correctness notes:

- Finished sequences never enter the aggregation: SGLang's continuous
  batching retires finished requests from the running batch, so every row
  of a decode ``ForwardBatch`` is an active sequence by construction.
- RadixAttention prefix cache: during extend the ``act_fn`` hook only sees
  the uncached suffix, so ``s0`` is computed over that suffix (accepted for
  Phase 1).
- Tensor parallelism: the intermediate dim is sharded across TP ranks and
  each rank's hook sees its local shard of ``Z``, so the top-k is per-shard
  ((1 - sparsity) of each rank's local neurons) with no cross-rank
  communication.
- Eager mode only: CUDA-graph replay does not run Python forward hooks, so
  masks only apply on the eager forward path. Hooks are registered after
  graph capture so no capture ever traces them.

Phase 2 (explicitly out of scope here): fused gather-GEMM kernels,
CUDA-graph capture of the dynamic mask, global cross-TP top-k.
"""

import enum
import logging
import math
from typing import Dict, List, Optional

import torch

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_EPS = 1e-8


class _StepMode(enum.Enum):
    """What the hooks should do with the current forward's activations."""

    IDLE = "idle"
    PROMPT = "prompt"  # prompt phase: collect s0, stay dense
    EXPLORE = "explore"  # dense-exploration decode step: collect scores
    PRUNE = "prune"  # sparse decode step: apply the shared mask


def _row_l2_normalize(z: torch.Tensor) -> torch.Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def compute_prompt_scores(z: torch.Tensor) -> torch.Tensor:
    """Importance score over a prompt (extend) forward, Eq. 2 pooled.

    Row-L2-normalize each token's activation vector, then take the column-L2
    over tokens divided by ``sqrt(T_valid)``. The extend batch is pooled
    across requests (Phase-1 simplification); padding/EOS tokens are absent
    because extend forwards only carry real tokens.
    """
    z_norm = _row_l2_normalize(z.float())
    return z_norm.norm(dim=0) / math.sqrt(max(z.shape[0], 1))


def compute_decode_scores(z: torch.Tensor) -> torch.Tensor:
    """Importance score over a decode forward: Eq. 2 per row, Eq. 3 batch max.

    Each decode row is one token of one active sequence, so the per-sequence
    score (T_valid=1) is the magnitude of its row-normalized activation, and
    the batch aggregation is the element-wise max over the active rows.
    """
    return _row_l2_normalize(z.float()).abs().amax(dim=0)


def build_topk_mask(scores: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Binary keep-mask of the top ``round((1 - sparsity) * D)`` neurons."""
    dim = scores.shape[0]
    k = min(dim, max(1, round((1.0 - sparsity) * dim)))
    mask = torch.zeros_like(scores)
    mask.scatter_(0, torch.topk(scores, k).indices, 1.0)
    return mask


def find_act_fn_hook_targets(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """``{module_name: module}`` for every gated-MLP ``SiluAndMul`` (``*.mlp.act_fn``)."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, SiluAndMul) and name.endswith("mlp.act_fn")
    }


class BWAPManager:
    """Holds per-layer BWAP state, the phase schedule, and the forward hooks.

    Mirrors the role of ``lora/lora_manager.py``: constructed once by the
    ``ModelRunner``, fed batch context via ``prepare_bwap_batch`` next to
    LoRA's ``prepare_lora_batch``, and acting on the model only through the
    forward hooks registered by ``register_hooks``.
    """

    def __init__(
        self,
        *,
        base_model: torch.nn.Module,
        sparsity: float = 0.5,
        t_init: int = 8,
        t_explore: int = 4,
        t_prune: int = 16,
    ):
        self.sparsity = sparsity
        self.t_init = t_init
        self.t_explore = t_explore
        self.t_prune = t_prune
        self.t_trans = t_explore + t_prune

        self.hook_targets = find_act_fn_hook_targets(base_model)
        if not self.hook_targets:
            logger.warning(
                "BWAP: no '*.mlp.act_fn' (SiluAndMul) modules found; "
                "--enable-bwap will have no effect on this model."
            )

        # Per-layer state, keyed by module name, lazily allocated on the
        # first hook call (the local intermediate size is only known then).
        self.mem_scores: Dict[str, torch.Tensor] = {}
        self.masks: Dict[str, torch.Tensor] = {}
        # Version counters let prune steps lazily rebuild a stale mask from
        # the latest max-aggregated scores (no cross-step scheduling logic).
        self.mem_version: Dict[str, int] = {}
        self.mask_version: Dict[str, int] = {}

        self.decode_step = 0
        self.num_dense_steps = 0
        self.num_prune_steps = 0
        self._mode = _StepMode.IDLE
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    @property
    def realized_sparsity(self) -> float:
        """Target sparsity discounted by dense steps (prompt + exploration),
        which are never pruned."""
        total = self.num_dense_steps + self.num_prune_steps
        if total == 0:
            return 0.0
        return self.sparsity * self.num_prune_steps / total

    def prepare_bwap_batch(self, forward_batch: ForwardBatch) -> None:
        """Set the step mode for the hooks from the batch's forward mode.

        Reads the batch only; all BWAP state lives on this manager. Called
        next to ``LoRAManager.prepare_lora_batch`` in ``ForwardBatch.init_new``.
        """
        forward_mode = forward_batch.forward_mode
        if forward_mode.is_extend():
            self._mode = _StepMode.PROMPT
            self.num_dense_steps += 1
        elif forward_mode.is_decode():
            self._mode = self._decode_step_mode(self.decode_step)
            self.decode_step += 1
            if self._mode is _StepMode.PRUNE:
                self.num_prune_steps += 1
            else:
                self.num_dense_steps += 1
        else:
            self._mode = _StepMode.IDLE

    def _decode_step_mode(self, step: int) -> _StepMode:
        if step < self.t_init:
            return _StepMode.EXPLORE
        cycle_pos = (step - self.t_init) % self.t_trans
        return _StepMode.PRUNE if cycle_pos < self.t_prune else _StepMode.EXPLORE

    def register_hooks(self) -> None:
        """Attach the post-forward hook to every target ``act_fn`` module.

        Must run after CUDA-graph capture so hook tensor ops are never
        traced into a captured graph (mirrors ``register_forward_hooks``).
        """
        for name, module in self.hook_targets.items():
            self._hook_handles.append(
                module.register_forward_hook(self._make_hook(name))
            )
        logger.info(
            "BWAP: registered pruning hooks on %d act_fn modules "
            "(sparsity=%.2f, T_init=%d, T_E=%d, T_p=%d).",
            len(self._hook_handles),
            self.sparsity,
            self.t_init,
            self.t_explore,
            self.t_prune,
        )

    def remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            return self._process_activation(name, output)

        return hook

    def _process_activation(
        self, name: str, z: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Collect scores from, or prune, one layer's activation ``Z``.

        Returning a value from a forward hook replaces the module output, so
        the masked ``Z`` flows into ``down_proj``.
        """
        if self._mode is _StepMode.PROMPT:
            self._update_mem(name, compute_prompt_scores(z))
            return None
        if self._mode is _StepMode.EXPLORE:
            self._update_mem(name, compute_decode_scores(z))
            return None
        if self._mode is _StepMode.PRUNE:
            mask = self._get_mask(name, z)
            return z * mask
        return None

    def _update_mem(self, name: str, scores: torch.Tensor) -> None:
        mem = self.mem_scores.get(name)
        self.mem_scores[name] = (
            scores.detach() if mem is None else torch.maximum(mem, scores)
        )
        self.mem_version[name] = self.mem_version.get(name, 0) + 1

    def _get_mask(self, name: str, z: torch.Tensor) -> torch.Tensor:
        dim = z.shape[-1]
        mem = self.mem_scores.get(name)
        stale = self.mask_version.get(name) != self.mem_version.get(name)
        if name not in self.masks or stale:
            if mem is None:
                # No scores collected yet (e.g. T_init=0 before any prompt):
                # an all-ones mask reproduces the dense output exactly.
                self.masks[name] = torch.ones(
                    dim, dtype=torch.float32, device=z.device
                )
            else:
                self.masks[name] = build_topk_mask(mem, self.sparsity)
            self.mask_version[name] = self.mem_version.get(name, 0)
        return self.masks[name].to(z.dtype)
