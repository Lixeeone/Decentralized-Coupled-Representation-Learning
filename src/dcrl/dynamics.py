"""Algorithm 1, with explicit optional operator ablations."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from .geometry import Domain
from .metrics import matrix, initialize_w, orthonormal_rows


@dataclass(frozen=True)
class DynamicsConfig:
    rank: int = 2
    eta_x: float = 0.01
    eta_w: float = 0.00005
    diffusion: float = 0.01
    beta: float = 1.0
    init_scale: float = 0.8
    inhibition: str = "matrix"
    spatial: bool = True
    plasticity: bool = True
    stabilize_every: int | None = None
    block_size: int = 2048
    divergence_norm: float = 1e8

    def __post_init__(self):
        if self.rank < 1 or self.block_size < 1:
            raise ValueError("rank and block_size must be positive")
        for key in ("eta_x", "eta_w", "init_scale", "divergence_norm"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if self.diffusion < 0 or self.beta < 0 or not np.isfinite([self.diffusion, self.beta]).all():
            raise ValueError("diffusion and beta must be nonnegative and finite")
        if self.inhibition not in {"matrix", "scalar", "none", "sanger"}:
            raise ValueError("unknown inhibition rule")
        if self.stabilize_every is not None and self.stabilize_every < 1:
            raise ValueError("stabilize_every must be positive or null")


def oja_delta(x, w, inhibition="matrix", block_size=2048):
    """Local accumulation, O(Nnm), with no n by n training covariance.

    Scalar ablation: E[||Wx||^2] W, as distinct from matrix lateral inhibition.
    Sanger is the triangular GHA reference, not the synchronous DCRL default.
    """
    delta = np.zeros_like(w)
    for start in range(0, len(x), block_size):
        b = x[start:start + block_size]
        y = b @ w.T
        if inhibition == "matrix":
            delta += y.T @ (b - y @ w)
        elif inhibition == "scalar":
            delta += y.T @ b - np.sum(y * y) * w
        elif inhibition == "none":
            delta += y.T @ b
        elif inhibition == "sanger":
            delta += y.T @ b - np.tril(y.T @ y) @ w
        else:
            raise ValueError("unknown inhibition")
    return delta / len(x)


def spatial_step(x, w, domain, cfg, rng):
    if not cfg.spatial:
        return x.copy()
    drift = domain.tangent(x, (x @ w.T) @ w)
    noise = domain.tangent(x, rng.normal(size=x.shape))
    return domain.retract(x + cfg.eta_x * cfg.diffusion * cfg.beta * drift
                          + np.sqrt(2 * cfg.diffusion * cfg.eta_x) * noise)


class DCRL:
    """Stateful synchronous solver. No hidden normalization or target feedback."""

    def __init__(self, x0, config: DynamicsConfig, domain: Domain, seed=0, w0=None):
        self.config, self.domain = config, domain
        self.x = matrix(x0).copy()
        if config.rank > self.x.shape[1]:
            raise ValueError("latent rank exceeds ambient dimension")
        if config.spatial and domain.kind == "identity":
            raise ValueError("active spatial dynamics require a bounded domain")
        self.w = (initialize_w(config.rank, self.x.shape[1], seed + 17, config.init_scale)
                  if w0 is None else matrix(w0, "w0").copy())
        if self.w.shape != (config.rank, self.x.shape[1]):
            raise ValueError("w0 shape mismatch")
        self.rng = np.random.default_rng(seed)
        self.step = 0

    def advance(self):
        c = self.config
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            x = spatial_step(self.x, self.w, self.domain, c, self.rng)
            w = self.w.copy()
            if c.plasticity:
                w += c.eta_w * oja_delta(x, w, c.inhibition, c.block_size)
            if c.stabilize_every and (self.step + 1) % c.stabilize_every == 0:
                w = orthonormal_rows(w)
            if not np.isfinite(x).all() or not np.isfinite(w).all() or np.linalg.norm(w) > c.divergence_norm:
                raise FloatingPointError("nonfinite state or configured divergence norm exceeded")
        self.x, self.w = x, w
        self.step += 1

    def state_dict(self):
        return {"step": self.step, "rng_state": self.rng.bit_generator.state,
                "dynamics": asdict(self.config)}
