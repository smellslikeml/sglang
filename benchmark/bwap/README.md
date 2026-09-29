# Batch-wise Adaptive Pruning (BWAP) — Phase-1 validation

BWAP (`--enable-bwap`, arXiv:2608.14003) is a training-free, inference-time FFN
neuron-pruning method for gated-MLP models, built for batched decode. This
directory documents how to validate the **Phase-1 functional integration**.

## What Phase 1 validates (and what it does not)

Phase 1 is **functional**: a post-forward hook on each `*.mlp.act_fn` masks
low-importance neurons of the activation (`Z * mask`) during sparse decode
steps. It validates:

- **Correctness** — an all-ones mask reproduces the dense output; the schedule
  and per-request phase gating behave as specified (unit tests,
  `test/registered/unit/bwap/test_bwap_manager.py`).
- **Accuracy retention** — pruned generation stays close to dense on a
  reasoning benchmark, and stays *flat as batch size grows* (BWAP's central
  claim vs. threshold methods that collapse under a shared batch mask).
- **Realized sparsity** — `BWAPManager.realized_sparsity` (target sparsity
  discounted by the dense prompt/exploration steps, which are never pruned).

Phase 1 does **not** measure throughput, and deliberately so: the masked path
(`Z * mask`) does the *same* GEMM as dense, so it saves no compute. The actual
speedup requires a **fused gather-GEMM** over the retained rows (adapting
`layers/moe/fused_moe_triton`) plus CUDA-graph capture of the dynamic mask —
that is Phase 2. Because CUDA-graph replay does not run Python forward hooks,
Phase-1 pruning only applies on the eager path (`--disable-decode-cuda-graph`),
so a Phase-1 dense-vs-pruned wall-clock comparison would be eager-vs-eager and
is not meaningful.

## Accuracy-retention protocol (GPU)

Reuses the existing GSM8K benchmark (`benchmark/gsm8k`). Run it against a dense
server and a BWAP-pruned server and compare accuracy; repeat at batch sizes 1
and 4+ to check batch-invariance.

```bash
# 1) Dense baseline
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 30000
python benchmark/gsm8k/bench_sglang.py --num-questions 200 --parallel 8

# 2) BWAP-pruned (eager path; hooks do not run under CUDA-graph replay)
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 30000 \
  --enable-bwap --bwap-sparsity 0.5 --disable-decode-cuda-graph
python benchmark/gsm8k/bench_sglang.py --num-questions 200 --parallel 8
```

Expectation (from arXiv:2608.14003, Table 2, DS-R1-Distill-Qwen-7B @ 50%
target sparsity, batch 4): the shared-mask method retains accuracy close to
dense where a TEAL-style threshold baseline collapses. An independent CPU
reference reproduction of the method (accuracy flat across batch 1→4, ~40%
realized sparsity) is in the companion notebooks; the schedule/mask math here
is unit-tested to match that reference.

`--bwap-t-init`, `--bwap-t-explore`, `--bwap-t-prune` tune the schedule
(defaults 8 / 4 / 16; `T_trans = T_E + T_p = 20`, near the paper's ~22-token
median neuron re-firing period).

## Unit tests (CPU)

```bash
python -m pytest test/registered/unit/bwap/test_bwap_manager.py -v
```

Covers Eq. 2/3 scoring, top-k mask selection, the per-request three-phase
schedule, the continuous-batching case (a late-joining request explores before
it is pruned), and the hook applying the shared mask only to in-cycle rows.
