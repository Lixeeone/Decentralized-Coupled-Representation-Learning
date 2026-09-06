"""Algorithm 2: local spatial update, local Oja, then active-edge mixing."""
from __future__ import annotations
import numpy as np
from .dynamics import oja_delta, spatial_step
from .metrics import matrix, initialize_w


def pairwise_mix(w_i, w_j, alpha):
    if not 0 < alpha <= 0.5:
        raise ValueError("alpha must be in (0,0.5]")
    # Independent RHS arrays avoid view aliasing when stored back in place.
    return ((1 - alpha) * w_i + alpha * w_j,
            alpha * w_i + (1 - alpha) * w_j)


class ADCRL:
    def __init__(self, x0, config, domain, seed=0, alpha=0.5, graph="ring"):
        self.x = matrix(x0).copy()
        self.config, self.domain = config, domain
        if len(self.x) < 2 or not 0 < alpha <= 0.5 or graph not in {"ring", "complete"}:
            raise ValueError("invalid population, gossip weight, or communication graph")
        if config.spatial and domain.kind == "identity":
            raise ValueError("active motion requires bounded retraction")
        if config.stabilize_every is not None:
            raise ValueError("A-DCRL does not support global QR stabilization")
        self.alpha, self.graph, self.step = alpha, graph, 0
        self.rng = np.random.default_rng(seed)
        base = initialize_w(config.rank, self.x.shape[1], seed + 31, config.init_scale)
        self.w_agents = np.repeat(base[None, :, :], len(self.x), axis=0)
        self.w_agents += 1e-3 * self.rng.normal(size=self.w_agents.shape)

    @property
    def w(self):
        """Centralized readout for evaluation only, never an update primitive."""
        return self.w_agents.mean(axis=0)

    def consensus_error(self):
        return float(np.sqrt(np.mean(np.sum((self.w_agents - self.w) ** 2, axis=(1, 2)))))

    def advance(self, edge=None):
        if edge is None:
            i = int(self.rng.integers(len(self.x)))
            if self.graph == "ring":
                j = (i + (1 if self.rng.random() < 0.5 else -1)) % len(self.x)
            else:
                j = (i + int(self.rng.integers(1, len(self.x)))) % len(self.x)
        else:
            i, j = edge
        if not (0 <= i < len(self.x) and 0 <= j < len(self.x)) or i == j:
            raise ValueError("invalid active edge")
        c = self.config
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            xi = spatial_step(self.x[i:i+1], self.w_agents[i], self.domain, c, self.rng)
            wi = self.w_agents[i].copy()
            if c.plasticity:
                wi += c.eta_w * oja_delta(xi, wi, c.inhibition)
            wi, wj = pairwise_mix(wi, self.w_agents[j], self.alpha)
            if not all(np.isfinite(a).all() for a in (xi, wi, wj)) or max(np.linalg.norm(wi), np.linalg.norm(wj)) > c.divergence_norm:
                raise FloatingPointError("active endpoint diverged")
        self.x[i], self.w_agents[i], self.w_agents[j] = xi[0], wi, wj
        self.step += 1
        return i, j
