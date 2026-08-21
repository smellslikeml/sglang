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
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.bwap.bwap_fused import (
    fast_path_eligible,
    fused_pruned_mlp,
    gather_ffn_weights,
)
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
        fused: bool = False,
        tp_size: int = 1,
        model_runner=None,
    ):
        self.sparsity = sparsity
        self.t_init = t_init
        self.t_explore = t_explore
        self.t_prune = t_prune
        self.t_trans = t_explore + t_prune
        self.fused = fused
        self.tp_size = tp_size
        # For Phase-2b recapture: reach the decode CUDA-graph runner to rebuild
        # the graph AFTER the mask is frozen (bakes the real weights at capture,
        # sidestepping the post-capture-update-visibility issue).
        self._model_runner = model_runner
        self._recapture_pending = False

        self.hook_targets = find_act_fn_hook_targets(base_model)
        if not self.hook_targets:
            logger.warning(
                "BWAP: no '*.mlp.act_fn' (SiluAndMul) modules found; "
                "--enable-bwap will have no effect on this model."
            )
        # For Phase-2a: the gated-MLP module parent of each act_fn (keyed by the
        # same act_fn name as the mem/mask state), so the fused forward can gather
        # the layer's gate_up/down weights.
        self.gated_mlps: Dict[str, torch.nn.Module] = {}
        if fused:
            for name in self.hook_targets:
                self.gated_mlps[name] = base_model.get_submodule(name.rsplit(".", 1)[0])

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
        self._all_prune = False  # every active row is pruning -> fast path eligible

        # Phase-2a fused-forward state (keyed by act_fn name).
        self._fast_ok: Dict[str, bool] = {}  # layer eligible for the gather path
        # Fixed-address gathered-weight buffers (allocated once at the fixed size
        # k=(1-sparsity)*D_ff, refreshed in place with copy_). Stable pointers so a
        # captured CUDA graph reads the same address every replay (Phase 2b).
        self._gate_up_buf: Dict[str, torch.Tensor] = {}  # [2k, hidden] gate+up rows
        self._down_buf: Dict[str, torch.Tensor] = {}  # [hidden, k] down columns
        self._gathered_version: Dict[str, int] = {}  # mask_version the buffers reflect
        self._orig_forwards: Dict[str, Callable] = {}  # saved mlp.forward for teardown
        # Phase-2b throughput probe: when set (only around decode-graph capture),
        # the wrapper unconditionally takes the gather path over pre-filled dummy
        # buffers, so the captured decode graph is the k-width pruned FFN. Correctness
        # is ignored (dummy mask) — this measures the end-to-end speedup ceiling.
        self._capture_force = False

        # Phase-2b (correct): frozen-mask-after-warmup. can_run_graph() gates the
        # captured pruned graph on _graph_ready; while False the runner stays eager
        # so exploration hooks build the real mask. After `_warmup_steps` decode
        # forwards, _freeze_and_fill() fills the (fixed-address) buffers from the
        # real mask and opens the gate. Default True = ungated (probe / non-2b).
        self._graph_ready = True
        self._graph_gated = False  # armed only for real 2b (graph + fused, non-probe)
        self._decode_forwards = 0
        self._warmup_steps = t_init
        # Real (post-warmup) gathered weights, source for the registry post_fill
        # that copies them into the graph-resident buffers on the replay stream —
        # a bare copy_ in init_new isn't visible to the captured graph.
        self._real_gate_up: Dict[str, torch.Tensor] = {}
        self._real_down: Dict[str, torch.Tensor] = {}
        self._graph_buffers_registered = False

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
        # Phase-2b: once warmup froze the mask, rebuild the decode graph with the
        # real weights baked in (between forwards, before this batch runs).
        if self._recapture_pending:
            self.maybe_recapture()
        if forward_mode.is_extend():
            self._phase = _Phase.PROMPT
            for idx, seq_len in zip(req_pool_indices.tolist(), seq_lens.tolist()):
                self.prompt_lens[idx] = seq_len
            self._prune_rows = self._collect_rows = None
            self._has_prune = False
            self._has_collect = True
            self._all_prune = False
        elif forward_mode.is_decode():
            self._phase = _Phase.DECODE
            steps = self._decode_steps(req_pool_indices, seq_lens)
            self._prune_rows, self._collect_rows = compute_row_modes(
                steps, t_init=self.t_init, t_prune=self.t_prune, t_trans=self.t_trans
            )
            self._has_prune = bool(self._prune_rows.any())
            self._has_collect = bool(self._collect_rows.any())
            # Fast gather path applies only when the whole active batch is pruning
            # (a shared mask, no rows still exploring); mixed steps fall back to
            # the per-row Phase-1 masked path.
            self._all_prune = self._has_prune and not self._has_collect
            self._total_row_steps += int(steps.numel())
            self._pruned_row_steps += int(self._prune_rows.sum())
            # Phase-2b: after the eager warmup builds the real mask, freeze it into
            # the graph-resident buffers and open the graph gate.
            if self._graph_gated and not self._graph_ready:
                self._decode_forwards += 1
                if (
                    self._decode_forwards >= self._warmup_steps
                    and self._all_layers_have_scores()
                ):
                    self._freeze_and_fill()
        else:
            self._phase = _Phase.IDLE
            self._all_prune = False

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
        return self._materialize_mask(name, z.shape[-1], z.device).to(z.dtype)

    def _materialize_mask(
        self, name: str, dim: int, device: torch.device
    ) -> torch.Tensor:
        """Build/refresh the shared keep-mask for a layer from its scores.

        Split out from ``_get_mask`` so the fused forward can (re)build the mask
        without an activation tensor to read the shape/device from.
        """
        mem = self.mem_scores.get(name)
        stale = self.mask_version.get(name) != self.mem_version.get(name)
        if name not in self.masks or stale:
            if mem is None:
                # No scores collected yet (e.g. T_init=0 before any prompt): an
                # all-ones mask reproduces the dense output exactly.
                self.masks[name] = torch.ones(dim, dtype=torch.float32, device=device)
            else:
                self.masks[name] = build_topk_mask(mem, self.sparsity)
            self.mask_version[name] = self.mem_version.get(name, 0)
        return self.masks[name]

    # ---- Phase 2a: fused gather-GEMM forward ------------------------------
    def install_fused_forwards(self) -> None:
        """Wrap each gated MLP's ``forward`` with the fused fast path.

        Installed alongside ``register_hooks`` (both after CUDA-graph capture):
        the wrapper short-circuits to the gather-GEMM on all-prune decode steps
        of eligible layers, and otherwise calls the original dense forward — at
        which point the act_fn hook applies the Phase-1 collect/mask. The two are
        mutually exclusive per step, so no double-processing.
        """
        n_ok = 0
        for name, mlp in self.gated_mlps.items():
            self._fast_ok[name] = fast_path_eligible(
                mlp.gate_up_proj, mlp.down_proj, self.tp_size
            )
            n_ok += int(self._fast_ok[name])
            self._orig_forwards[name] = mlp.forward
            mlp.forward = self._make_mlp_forward(name, mlp)
        logger.info(
            "BWAP: fused gather-GEMM installed on %d/%d gated MLPs "
            "(others fall back to the masked path; tp_size=%d).",
            n_ok,
            len(self.gated_mlps),
            self.tp_size,
        )

    def remove_fused_forwards(self) -> None:
        for name, orig in self._orig_forwards.items():
            self.gated_mlps[name].forward = orig
        self._orig_forwards = {}

    def _make_mlp_forward(self, name: str, mlp: torch.nn.Module):
        orig = self._orig_forwards[name]

        def forward(x, *args, **kwargs):
            # Probe: force the gather path over pre-filled dummy buffers so the
            # captured decode graph is the pruned FFN (correctness ignored).
            if self._capture_force and self._fast_ok.get(name, False):
                return fused_pruned_mlp(
                    x, self._gate_up_buf[name], self._down_buf[name]
                )
            if (
                self._phase is _Phase.DECODE
                and self._all_prune
                and self._fast_ok.get(name, False)
                # Only fast-path once a real top-k mask exists: its support is
                # exactly k=(1-sparsity)*D_ff, which keeps the gathered buffers a
                # fixed size (the all-ones warmup mask would be full-width).
                and self.mem_scores.get(name) is not None
            ):
                return self._fast_mlp_forward(name, mlp, x)
            return orig(x, *args, **kwargs)

        return forward

    def begin_capture(self) -> None:
        """Phase-2b throughput probe: pre-fill fixed-size dummy gathered buffers
        (first k neurons) for every eligible layer and force the gather path, so
        the decode graph captured next runs the k-width pruned FFN. Buffers are
        allocated here (before capture) so no cudaMalloc happens during capture.
        Correctness is intentionally ignored — this measures the speedup ceiling.
        """
        for name, mlp in self.gated_mlps.items():
            if not self._fast_ok.get(name, False):
                continue
            weight = mlp.down_proj.weight
            d_ff = weight.shape[1]
            k = int(round((1.0 - self.sparsity) * d_ff))
            keep_idx = torch.arange(k, device=weight.device)  # dummy: first k
            gate_up_k, down_k = gather_ffn_weights(
                mlp.gate_up_proj.weight, weight, keep_idx, d_ff
            )
            self._gate_up_buf[name] = gate_up_k.contiguous()
            self._down_buf[name] = down_k.contiguous()
        self._capture_force = True
        if self._gate_up_buf:
            _k0 = next(iter(self._gate_up_buf))
            logger.info(
                "BWAP capture: gate_up_buf[%s] ptr=%x",
                _k0,
                self._gate_up_buf[_k0].data_ptr(),
            )

    def end_capture(self) -> None:
        self._capture_force = False

    # ---- Phase 2b (correct): frozen-mask-after-warmup graph gating ----------
    def arm_graph_gating(self) -> None:
        """Gate the captured pruned graph until warmup builds the real mask.

        Called at setup for real 2b (fused + decode graph, not the probe). Until
        `_freeze_and_fill` runs, `can_run_graph` sees `graph_ready()==False` and
        keeps decode eager so the act_fn hooks collect scores.
        """
        self._graph_gated = True
        self._graph_ready = False

    def graph_ready(self) -> bool:
        return self._graph_ready

    def _all_layers_have_scores(self) -> bool:
        return all(
            self.mem_scores.get(name) is not None
            for name in self.gated_mlps
            if self._fast_ok.get(name, False)
        )

    def _freeze_and_fill(self) -> None:
        """Freeze the warmed-up mask: stash the real gathered weights as the source
        the registry post_fill copies into the graph buffers each replay, and open
        the gate. (The captured graph reads _gate_up_buf/_down_buf; a bare copy_ into
        them from here isn't visible on replay — post_fill on the replay stream is.)"""
        for name, mlp in self.gated_mlps.items():
            if not self._fast_ok.get(name, False):
                continue
            weight = mlp.down_proj.weight
            self._materialize_mask(name, weight.shape[1], weight.device)
            keep_idx = self.masks[name].bool().nonzero(as_tuple=False).squeeze(-1)
            gate_up_k, down_k = gather_ffn_weights(
                mlp.gate_up_proj.weight, weight, keep_idx, weight.shape[1]
            )
            self._real_gate_up[name] = gate_up_k.contiguous()
            self._real_down[name] = down_k.contiguous()
            # Also fill the graph buffers now so the eager path (graphs disabled)
            # and the first post-freeze replay are correct even before a post_fill.
            if name in self._gate_up_buf:
                self._gate_up_buf[name].copy_(self._real_gate_up[name])
                self._down_buf[name].copy_(self._real_down[name])
        # Request a recapture: the graph stays gated (eager) until the decode graph
        # is rebuilt with these real weights baked in (fill-per-replay proved not
        # visible in full-model capture; recapture bakes them at capture time).
        self._recapture_pending = self._model_runner is not None
        if not self._recapture_pending:
            self._graph_ready = True  # no runner (tests): fall back to fill path
        logger.info(
            "BWAP: warmup complete (%d decode forwards) — froze mask; recapture_pending=%s.",
            self._decode_forwards,
            self._recapture_pending,
        )

    def maybe_recapture(self) -> None:
        """Rebuild the decode CUDA graph with the frozen real weights baked in.
        Triggered between forwards (top of prepare_bwap_batch) once, after freeze.
        _capture_force makes the wrapper take the gather path during recapture, and
        the buffers already hold the real weights, so the new graph reads them."""
        if not self._recapture_pending:
            return
        self._recapture_pending = False
        runner = self._model_runner.decode_cuda_graph_runner
        if runner is None:
            self._graph_ready = True  # nothing to recapture; use eager fill fallback
            return
        logger.info("BWAP: recapturing decode graph with frozen weights ...")
        self._capture_force = True
        try:
            runner.capture()
        finally:
            self._capture_force = False
        self._graph_ready = True
        logger.info("BWAP: recapture complete — graph gate opened.")

    def register_graph_buffers(self, registry) -> None:
        """Bind the gathered-weight buffers into SGLang's CUDA-graph buffer registry
        with a post_fill that copies the frozen real weights in on the replay stream.
        This is what makes post-capture buffer updates visible to the replayed graph
        (a bare copy_ is not). Called once from the decode runner before capture."""
        if self._graph_buffers_registered:
            return
        from sglang.srt.model_executor.cuda_graph_buffer_registry import GraphSlot

        self._post_fill_logged = False

        def _make_post_fill(src: Dict[str, torch.Tensor], key: str):
            # No-op until warmup fills `src`; then copy the frozen weights into the
            # graph buffer on the replay stream (what makes them visible on replay).
            def fill(buffer, forward_batch, ctx):
                real = src.get(key)
                if not self._post_fill_logged:
                    logger.info(
                        "BWAP post_fill: real_present=%s buf_ptr=%x shape=%s",
                        real is not None,
                        buffer.data_ptr(),
                        tuple(buffer.shape),
                    )
                    self._post_fill_logged = True
                if real is not None:
                    buffer.copy_(real)

            return fill

        for name in self.gated_mlps:
            if not self._fast_ok.get(name, False):
                continue
            for tag, buf, src in (
                ("gate_up", self._gate_up_buf[name], self._real_gate_up),
                ("down", self._down_buf[name], self._real_down),
            ):
                shape = tuple(buf.shape)
                registry.register_slot(
                    GraphSlot(
                        name=f"bwap.{tag}.{name}",
                        shape_fn=lambda mb, mt, s=shape: s,
                        dtype=buf.dtype,
                        device=buf.device,
                        axis="none",  # fixed weight buffers, NOT token-indexed — don't slice
                        copy_from_fb=False,
                        post_fill=_make_post_fill(src, name),
                    ),
                    bind=buf,
                )
        self._graph_buffers_registered = True
        logger.info(
            "BWAP: registered %d graph-resident buffers.", 2 * len(self._gate_up_buf)
        )

    def _fast_mlp_forward(
        self, name: str, mlp: torch.nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        weight = mlp.down_proj.weight
        d_ff = weight.shape[1]
        self._materialize_mask(name, d_ff, weight.device)
        if self._gathered_version.get(name) != self.mask_version.get(name):
            keep_idx = self.masks[name].bool().nonzero(as_tuple=False).squeeze(-1)
            gate_up_k, down_k = gather_ffn_weights(
                mlp.gate_up_proj.weight, weight, keep_idx, d_ff
            )
            if name not in self._gate_up_buf:  # allocate fixed-address buffers once
                self._gate_up_buf[name] = torch.empty_like(gate_up_k, dtype=x.dtype)
                self._down_buf[name] = torch.empty_like(down_k, dtype=x.dtype)
            self._gate_up_buf[name].copy_(gate_up_k)  # refresh in place (stable ptr)
            self._down_buf[name].copy_(down_k)
            self._gathered_version[name] = self.mask_version.get(name, 0)
        return fused_pruned_mlp(x, self._gate_up_buf[name], self._down_buf[name])
