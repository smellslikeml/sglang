"""Temperature-decoupled proxy scores for stochastic draft-tree construction.

Adapted from RheoSampling: *Resolving the One-Hot Dilemma in Stochastic
Dynamic-Tree Speculative Decoding* (arXiv:2609.21827).

Dynamic-tree EAGLE-style drafting builds and prunes its tree with a single
score — the running product of draft probabilities along each path — and reuses
that same distribution for verification. Under stochastic decoding (T>0) this
coupling collapses the draft distribution toward the greedy chain: because the
score is a product of probabilities in ``[0, 1]``, it shrinks with depth, so the
global top-k that prunes the tree systematically favors shallow high-probability
nodes and prunes the deeper branches that stochastic verification would actually
accept — dropping the acceptance rate.

RheoSampling's core idea is to *decouple the two roles*: give tree construction
and pruning a proxy probability while verification keeps the true sampling
probability. This module delivers that decoupling at SGLang's existing dynamic
tree call site (``organize_draft_results``): the global top-k selects nodes on a
temperature-gated, depth-normalized proxy score, while the tokens that flow
downstream to the verify kernels — and therefore the verification probabilities —
are untouched. Because only which candidates get drafted changes (never how they
are verified), speculative decoding stays lossless regardless of the proxy.

The proxy is a per-node concave power transform whose exponent interpolates,
by temperature, between the raw product (greedy) and the depth-normalized
geometric mean of per-step probabilities::

    t_frac   = T / (1 + T)                       # 0 at T=0, saturates toward 1
    exponent = 1 - t_frac * (1 - 1 / depth)      # 1 at depth 1 or T=0, ->1/depth hot
    proxy    = score ** exponent

At ``T = 0`` the exponent is exactly ``1`` for every node, so the greedy path is
a per-row no-op. As temperature rises, deeper nodes are boosted relative to the
greedy chain, keeping a diverse candidate set that better matches the flatter
stochastic verification distribution.

Scoping (Mode 2, adapted port): the paper's mechanism kept at full fidelity is
the construction/verification probability split. Its auxiliary components —
injecting a freshly multinomial-sampled token into the top-k slots, the OT-based
verification strategy, the sparse-draft mechanism, and the equivalence-class
losslessness proof — are intentionally out of scope; they would require a second
probability channel threaded through the verify kernels, which this call site
does not expose. Here the proxy is a parameter-free, depth-normalized transform
over the *existing* deterministic top-k scores.
"""

from __future__ import annotations

import math
from typing import List

import torch

from sglang.srt.environ import envs


def proxy_scores_enabled() -> bool:
    """Whether the RheoSampling proxy-score decoupling is turned on."""
    return envs.SGLANG_ENABLE_RHEO_PROXY_SCORES.get()


def draft_node_depths(score_list: List[torch.Tensor]) -> torch.Tensor:
    """Tree depth (1-indexed) of each flattened draft node.

    ``score_list`` is the per-step list assembled by the draft loop before it is
    concatenated and flattened: element ``j`` holds the scores produced at draft
    step ``j``, so every node it contributes sits at depth ``j + 1``. The number
    of nodes an element contributes per request is the product of its non-batch
    dimensions (it becomes that many columns after ``flatten(1)``).

    Returns a ``(num_nodes,)`` float tensor aligned with the flattened score
    columns, on the same device as the scores.
    """
    widths = [math.prod(step.shape[1:]) for step in score_list]
    device = score_list[0].device
    return torch.cat(
        [
            torch.full((width,), depth + 1, dtype=torch.float32, device=device)
            for depth, width in enumerate(widths)
        ]
    )


def temperature_proxy_scores(
    scores: torch.Tensor,
    temperatures: torch.Tensor,
    depths: torch.Tensor,
) -> torch.Tensor:
    """Proxy scores for draft-tree pruning, decoupled from verification.

    Args:
        scores: ``(bs, num_nodes)`` flattened draft-tree scores fed to the global
            top-k (running product of draft probabilities per candidate node).
        temperatures: ``(bs, 1)`` per-request sampling temperature. Broadcasts
            across the node dimension.
        depths: ``(num_nodes,)`` 1-indexed tree depth of each node, from
            :func:`draft_node_depths`.

    Returns:
        ``(bs, num_nodes)`` proxy scores used only to *rank* candidates. Rows with
        ``temperature == 0`` are exactly equal to ``scores`` (greedy untouched);
        the returned scores never reach verification.
    """
    # t_frac in [0, 1): 0 at T=0, saturating toward 1 as T grows.
    t_frac = temperatures / (1.0 + temperatures)
    # exponent = 1 at depth 1 or T=0; interpolates toward 1/depth as T grows.
    exponent = 1.0 - t_frac * (1.0 - 1.0 / depths)
    # clamp guards against tiny negative dust from upstream fused ops; scores are
    # semantically non-negative probabilities.
    proxy = scores.clamp_min(0.0).pow(exponent)
    # Keep greedy (T=0) rows bit-exact rather than relying on pow(x, 1.0) == x.
    return torch.where(temperatures > 0, proxy, scores)
