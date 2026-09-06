"""Asynchronous pairwise-gossip A-DCRL reference implementation.

Each agent maintains a local representation matrix. At each round, one active
edge is selected, one endpoint performs a local Oja update, and the two endpoint
matrices are mixed by pairwise averaging. Inactive agents are unchanged.
"""

from __future__ import annotations

import numpy as np

from .utils import initialize_w, make_rng, require_finite


def _ring_edge(num_agents: int, rng: np.random.Generator) -> tuple[int, int]:
    """Sample one active edge from a ring communication graph."""
    i = int(rng.integers(0, num_agents))
    if rng.random() < 0.5:
        j = (i - 1) % num_agents
    else:
        j = (i + 1) % num_agents
    return i, j


def _local_oja_delta(x_i: np.ndarray, w_i: np.ndarray) -> np.ndarray:
    """Single-sample subspace Oja update for one local matrix."""
    y_i = w_i @ x_i
    hebbian = np.outer(y_i, x_i)
    lateral = np.outer(y_i, y_i) @ w_i
    return hebbian - lateral


def run_adcrl(
    x0: np.ndarray,
    rank: int,
    num_rounds: int = 800,
    eta_w: float = 5.0e-4,
    gossip_alpha: float = 0.5,
    graph_type: str = "ring",
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run asynchronous pairwise-gossip A-DCRL.

    This compact implementation focuses on the representation update and
    communication footprint. The spatial configuration is kept fixed so that the
    routine isolates the local Oja-plus-gossip mechanism.
    """
    if graph_type != "ring":
        raise ValueError("This reference implementation currently supports graph_type='ring'.")

    x = np.asarray(x0, dtype=np.float64).copy()
    if x.ndim != 2:
        raise ValueError("x0 must have shape (N, n).")

    num_agents, dim = x.shape
    if num_agents < 2:
        raise ValueError("A-DCRL requires at least two agents for pairwise gossip.")
    if rank <= 0 or rank > dim:
        raise ValueError("rank must satisfy 0 < rank <= n.")
    if not (0.0 < gossip_alpha <= 0.5):
        raise ValueError("gossip_alpha must satisfy 0 < gossip_alpha <= 0.5.")

    rng = make_rng(seed)

    base_w = initialize_w(rank=rank, dim=dim, seed=seed + 31)
    local_w = np.repeat(base_w[None, :, :], repeats=num_agents, axis=0)

    # Small random perturbation avoids identical local trajectories.
    local_w += 1.0e-3 * rng.normal(size=local_w.shape)

    for _ in range(num_rounds):
        i, j = _ring_edge(num_agents, rng)

        w_i = local_w[i]
        w_j = local_w[j]

        # Local update at the active endpoint only.
        w_i_updated = w_i + eta_w * _local_oja_delta(x[i], w_i)

        # Pairwise active-edge gossip. No inactive agent is accessed or changed.
        local_w[i] = (1.0 - gossip_alpha) * w_i_updated + gossip_alpha * w_j
        local_w[j] = gossip_alpha * w_i_updated + (1.0 - gossip_alpha) * w_j

        require_finite("local_w", local_w)

    w_mean = local_w.mean(axis=0)

    return {
        "x": x,
        "w_agents": local_w,
        "w_mean": w_mean,
    }
