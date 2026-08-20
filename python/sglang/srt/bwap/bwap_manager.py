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
top-``k`` (``k = round((1 - sparsity) * D_FF)``) via a shared per-layer binary
mask, and zeroes the rest: ``Z * mask`` (functional; numerically identical to
gather-based pruning, without fused kernels).

Per-request phase schedule (defaults ``T_init=8, T_E=4, T_p=16``):

- Prompt (extend) forward: collect the prompt score ``s0``; stay dense.
- The first ``T_init`` decode steps *of each request*: dense exploration,
  updating the running max-aggregated score ``m``.
- Thereafter, cycles of ``T_trans = T_E + T_p`` steps: ``T_p`` sparse steps
  (apply the mask) then ``T_E`` dense-explore steps (refresh ``m``).

The step count is **per request**, derived from ``seq_lens - prompt_len``, not
a single global counter. This matters under continuous batching: a request that
joins a running batch mid-cycle still gets its own ``T_init`` exploration before
it is ever pruned, rather than being pruned immediately with a mask built from
other requests' activations. Score aggregation and mask application are gated
per batch row by that request's own phase, so a decode forward may collect from
some rows (in exploration) while pruning others (in a sparse phase). For a
*synchronized* batch (all requests submitted together) every row shares a phase,
and the schedule reduces exactly to the paper's global one.

Correctness notes:

- The mask stays shared across the batch (Eq. 3, element-wise max over the
  rows currently exploring); per-request gating only decides which rows
  contribute a score and which rows have the shared mask applied this step.
- Decode activation rows are assumed aligned with ``req_pool_indices`` /
  ``seq_lens`` order (one token per request; speculative decode, which packs
  multiple tokens per request, is left dense by a row-count guard).
- Finished sequences are retired from the running batch by SGLang's scheduler,
  so they never enter the aggregation.
- RadixAttention prefix cache: during extend the ``act_fn`` hook only sees the
  uncached suffix, so ``s0`` is computed over that suffix (accepted for Phase 1).
- Tensor parallelism: the intermediate dim is sharded across TP ranks and each
  rank's hook sees its local shard of ``Z``, so the top-k is per-shard
  ((1 - sparsity) of each rank's local neurons) with no cross-rank communication.
- Eager mode only: CUDA-graph replay does not run Python forward hooks, so masks
  only apply on the eager forward path. Hooks are registered after graph capture
  so no capture ever traces them.

Phase 2 (explicitly out of scope here): fused gather-GEMM kernels (the actual
throughput win — ``Z * mask`` saves no compute), CUDA-graph capture of the
dynamic mask, global cross-TP top-k. Phase 1 validates correctness,
accuracy-retention, and realized sparsity, not throughput.
"""

import enum
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)

_EPS = 1e-8


class _Phase(enum.Enum):
    """The current forward's phase; per-row exploration/pruning is derived from
    each request's own step within a DECODE forward."""

    IDLE = "idle"
    PROMPT = "prompt"  # extend forward: collect s0, stay dense
    DECODE = "decode"  # decode forward: collect and/or prune, gated per row


def _row_l2_normalize(z: torch.Tensor) -> torch.Tensor:
    return z / z.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def compute_prompt_scores(z: torch.Tensor) -> torch.Tensor:
    """Importance score over a prompt (extend) forward, Eq. 2 pooled.

    Row-L2-normalize each token's activation vector, then take the column-L2
    over tokens divided by ``sqrt(T_valid)``. The extend batch is pooled across
    requests (Phase-1 simplification); padding/EOS tokens are absent because
    extend forwards only carry real tokens.
    """
    z_norm = _row_l2_normalize(z.float())
    return z_norm.norm(dim=0) / math.sqrt(max(z.shape[0], 1))


def compute_decode_scores(z: torch.Tensor) -> torch.Tensor:
    """Importance score over decode rows: Eq. 2 per row, Eq. 3 batch max.

    Each row passed here is one token of one *exploring* active sequence, so the
    per-sequence score (T_valid=1) is the magnitude of its row-normalized
    activation, and the batch aggregation is the element-wise max over the rows.
    """
    return _row_l2_normalize(z.float()).abs().amax(dim=0)


def build_topk_mask(scores: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Binary keep-mask of the top ``round((1 - sparsity) * D)`` neurons."""
    dim = scores.shape[0]
    k = min(dim, max(1, round((1.0 - sparsity) * dim)))
    mask = torch.zeros_like(scores)
    mask.scatter_(0, torch.topk(scores, k).indices, 1.0)
    return mask


def compute_row_modes(
    steps: torch.Tensor, *, t_init: int, t_prune: int, t_trans: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-request (prune_rows, collect_rows) boolean masks from decode steps.

    A request explores (collects, no prune) for its first ``t_init`` steps, then
    alternates ``t_prune`` sparse steps and ``t_trans - t_prune`` explore steps.
    ``collect_rows`` is exactly the complement of ``prune_rows``: a row either
    contributes to the shared score this step or has the mask applied to it.
    """
    in_init = steps < t_init
    cycle_pos = (
        steps - t_init
    ) % t_trans  # torch int % is non-negative; masked when in_init
    prune_rows = (~in_init) & (cycle_pos < t_prune)
    return prune_rows, ~prune_rows


def find_act_fn_hook_targets(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """``{module_name: module}`` for every gated-MLP ``SiluAndMul`` (``*.mlp.act_fn``)."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, SiluAndMul) and name.endswith("mlp.act_fn")
    }


class BWAPManager:
    """Holds per-layer BWAP state, the per-request schedule, and the forward hooks.

    Mirrors the role of ``lora/lora_manager.py``: constructed once by the
    ``ModelRunner``, fed batch context via ``prepare_bwap_batch`` next to LoRA's
    ``prepare_lora_batch``, and acting on the model only through the forward
    hooks registered by ``register_hooks``.
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

        # Per-layer state, keyed by module name, lazily allocated on the first
        # hook call (the local intermediate size is only known then).
        self.mem_scores: Dict[str, torch.Tensor] = {}
        self.masks: Dict[str, torch.Tensor] = {}
        # Version counters let prune steps lazily rebuild a stale mask from the
        # latest max-aggregated scores (no cross-step scheduling logic).
        self.mem_version: Dict[str, int] = {}
        self.mask_version: Dict[str, int] = {}

        # Per-request prompt lengths, keyed by req_pool_index. Overwritten on
        # every extend forward, which is self-cleaning: chunked prefill converges
        # to the full prompt length by the first decode, and a reused pool slot
        # is reset by the new request's own extend.
        self.prompt_lens: Dict[int, int] = {}

        # Current-forward state, set by prepare_bwap_batch and read by the hooks.
        self._phase = _Phase.IDLE
        self._prune_rows: Optional[torch.Tensor] = None
        self._collect_rows: Optional[torch.Tensor] = None
        self._has_prune = False
        self._has_collect = False

        # Row-step accounting for realized_sparsity (a row-step is one request
        # advanced one decode step; prompt/exploration row-steps are never pruned).
        self._pruned_row_steps = 0
        self._total_row_steps = 0

        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    @property
    def realized_sparsity(self) -> float:
        """Target sparsity discounted by the fraction of row-steps actually
        pruned (exploration + prompt row-steps stay dense)."""
        if self._total_row_steps == 0:
            return 0.0
        return self.sparsity * self._pruned_row_steps / self._total_row_steps

    def prepare_bwap_batch(
        self,
        *,
        forward_mode: ForwardMode,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Set the per-row phase for the hooks from this forward's batch.

        Reads only the specific batch fields it needs; all BWAP state lives on
        this manager. Called next to ``LoRAManager.prepare_lora_batch`` in
        ``ForwardBatch.init_new``.
        """
        if forward_mode.is_extend():
            self._phase = _Phase.PROMPT
            for idx, seq_len in zip(req_pool_indices.tolist(), seq_lens.tolist()):
                self.prompt_lens[idx] = seq_len
            self._prune_rows = self._collect_rows = None
            self._has_prune = False
            self._has_collect = True
        elif forward_mode.is_decode():
            self._phase = _Phase.DECODE
            steps = self._decode_steps(req_pool_indices, seq_lens)
            self._prune_rows, self._collect_rows = compute_row_modes(
                steps, t_init=self.t_init, t_prune=self.t_prune, t_trans=self.t_trans
            )
            self._has_prune = bool(self._prune_rows.any())
            self._has_collect = bool(self._collect_rows.any())
            self._total_row_steps += int(steps.numel())
            self._pruned_row_steps += int(self._prune_rows.sum())
        else:
            self._phase = _Phase.IDLE

    def _decode_steps(
        self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        """Per-row decode step = ``seq_lens - prompt_len`` (clamped at 0)."""
        prompt_lens = torch.tensor(
            [
                self.prompt_lens.get(idx, seq_len)
                for idx, seq_len in zip(req_pool_indices.tolist(), seq_lens.tolist())
            ],
            device=seq_lens.device,
            dtype=seq_lens.dtype,
        )
        return (seq_lens - prompt_lens).clamp_min_(0)

    def register_hooks(self) -> None:
        """Attach the post-forward hook to every target ``act_fn`` module.

        Must run after CUDA-graph capture so hook tensor ops are never traced
        into a captured graph (mirrors ``register_forward_hooks``).
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

    def _process_activation(self, name: str, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Collect scores from, and/or prune, one layer's activation ``Z``.

        Returning a value from a forward hook replaces the module output, so the
        (partially) masked ``Z`` flows into ``down_proj``.
        """
        if self._phase is _Phase.PROMPT:
            self._update_mem(name, compute_prompt_scores(z))
            return None
        if self._phase is _Phase.DECODE:
            if self._prune_rows is None or z.shape[0] != self._prune_rows.shape[0]:
                # Unexpected row layout (e.g. speculative decode packs multiple
                # tokens per request); leave the activation dense this step.
                return None
            if self._has_collect:
                self._update_mem(name, compute_decode_scores(z[self._collect_rows]))
            if self._has_prune:
                mask = self._get_mask(name, z)
                gate = self._prune_rows.view(-1, *([1] * (z.dim() - 1)))
                return torch.where(gate, z * mask, z)
            return None
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
                # No scores collected yet (e.g. T_init=0 before any prompt): an
                # all-ones mask reproduces the dense output exactly.
                self.masks[name] = torch.ones(dim, dtype=torch.float32, device=z.device)
            else:
                self.masks[name] = build_topk_mask(mem, self.sparsity)
            self.mask_version[name] = self.mem_version.get(name, 0)
        return self.masks[name].to(z.dtype)
