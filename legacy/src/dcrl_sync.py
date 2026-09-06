"""Synchronous DCRL reference implementation.

This module implements the two-phase update used in Algorithm 1:

1. projected spatial step under the frozen representation-induced potential;
2. finite-sample subspace Oja update with lateral inhibition.
"""

from __future__ import annotations

import numpy as np

from .utils import (
    initialize_w,
    make_rng,
    project_rows_to_ball,
    require_finite,
    row_orthonormalize,
)


def _subspace_oja_delta(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Finite-sample subspace Oja drift with lateral inhibition.

    For Y = X W^T, the update is

        Delta W = mean_i [ y_i x_i^T - y_i y_i^T W ].

    Equivalently, Delta W = Y^T X / N - (Y^T Y / N) W.
    """
    y = x @ w.T
    n_samples = max(1, x.shape[0])

    hebbian = (y.T @ x) / n_samples
    lateral = ((y.T @ y) / n_samples) @ w

    return hebbian - lateral


def run_dcrl(
    x0: np.ndarray,
    rank: int,
    num_steps: int = 500,
    eta_x: float = 1.0e-3,
    eta_w: float = 3.0e-4,
    diffusion: float = 5.0e-3,
    beta: float = 1.0,
    seed: int = 0,
    domain_radius: float = 5.0,
    stabilize_every: int | None = None,
) -> dict[str, np.ndarray]:
    """Run synchronous DCRL on a compact reference substrate.

    Parameters
    ----------
    x0:
        Initial configuration matrix with shape (N, n).
    rank:
        Latent dimension m.
    num_steps:
        Number of discrete DCRL steps.
    eta_x, eta_w:
        Spatial and representation step sizes, with eta_w << eta_x.
    diffusion:
        Diffusion coefficient D.
    beta:
        Inverse-temperature coefficient for the spatial drift.
    seed:
        Random seed.
    domain_radius:
        Radius for the compact substrate-domain retraction Pi_X.
    stabilize_every:
        Optional frequency for numerical row-space stabilization. The default
        is None, so the reported reference update follows the finite-step Oja
        dynamics directly.

    Returns
    -------
    dict
        Contains final configuration ``x`` and final representation ``w``.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    if x.ndim != 2:
        raise ValueError("x0 must have shape (N, n).")

    _, dim = x.shape
    if rank <= 0 or rank > dim:
        raise ValueError("rank must satisfy 0 < rank <= n.")
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative.")
    if eta_x <= 0 or eta_w <= 0:
        raise ValueError("eta_x and eta_w must be positive.")
    if diffusion < 0:
        raise ValueError("diffusion must be non-negative.")
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    if domain_radius <= 0:
        raise ValueError("domain_radius must be positive.")
    if stabilize_every is not None and stabilize_every <= 0:
        raise ValueError("stabilize_every must be positive or None.")

    rng = make_rng(seed)
    w = initialize_w(rank=rank, dim=dim, seed=seed + 17)

    sqrt_noise_scale = np.sqrt(2.0 * diffusion * eta_x)

    for step in range(num_steps):
        # Phase 1: fast kinematics.
        # For U_W(x) = -0.5 ||W x||^2, the negative gradient direction is
        # W^T W x. In row-vector form this is x W^T W.
        spatial_drift = x @ w.T @ w
        noise = rng.normal(size=x.shape)

        x = x + eta_x * diffusion * beta * spatial_drift + sqrt_noise_scale * noise
        x = project_rows_to_ball(x, radius=domain_radius)

        # Phase 2: slow plasticity with lateral inhibition.
        w = w + eta_w * _subspace_oja_delta(x, w)

        # Optional numerical stabilization for finite-step reference runs.
        if stabilize_every is not None and (step + 1) % stabilize_every == 0:
            w = row_orthonormalize(w)

        require_finite("x", x)
        require_finite("w", w)

    return {
        "x": x,
        "w": w,
    }
