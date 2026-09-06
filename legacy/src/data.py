"""Self-contained reference substrates for the anonymous DCRL package.

The data generators are intentionally lightweight and self-contained. They are
used only for reference diagnostics; no external datasets are required.
"""

from __future__ import annotations

import numpy as np


def standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Center and scale a feature matrix.

    Parameters
    ----------
    x:
        Input matrix of shape (N, n).
    eps:
        Numerical floor for the global standard deviation.

    Returns
    -------
    np.ndarray
        Standardized matrix with approximately unit global scale.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must have shape (N, n).")

    x = x - x.mean(axis=0, keepdims=True)
    scale = float(np.std(x))
    return x / max(scale, eps)


def make_swiss_roll(
    n_samples: int,
    seed: int = 0,
    noise: float = 0.02,
) -> np.ndarray:
    """Generate a compact Swiss-roll reference substrate in R^3."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(seed)

    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n_samples))
    h = 2.0 * rng.random(n_samples) - 1.0

    x = np.stack(
        [
            t * np.cos(t),
            6.0 * h,
            t * np.sin(t),
        ],
        axis=1,
    )

    if noise > 0:
        x = x + noise * rng.normal(size=x.shape)

    return x


def make_sphere(
    n_samples: int,
    seed: int = 0,
    noise: float = 0.0,
) -> np.ndarray:
    """Generate a unit-sphere reference substrate in R^3."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(seed)

    x = rng.normal(size=(n_samples, 3))
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    if noise > 0:
        x = x + noise * rng.normal(size=x.shape)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    return x


def make_hd_gmm(
    n_samples: int,
    dim: int,
    num_clusters: int = 5,
    seed: int = 0,
    intrinsic_dim: int | None = None,
    cluster_scale: float = 4.0,
    noise: float = 0.35,
) -> np.ndarray:
    """Generate a controlled high-dimensional Gaussian-mixture substrate.

    The mixture has low-dimensional cluster structure embedded in an ambient
    feature space. This provides a compact reference input for high-dimensional
    diagnostic runs.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if dim < 2:
        raise ValueError("dim must be at least 2.")
    if num_clusters < 2:
        raise ValueError("num_clusters must be at least 2.")
    if intrinsic_dim is not None and not (1 <= intrinsic_dim <= dim):
        raise ValueError("intrinsic_dim must satisfy 1 <= intrinsic_dim <= dim.")
    if cluster_scale <= 0:
        raise ValueError("cluster_scale must be positive.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    rng = np.random.default_rng(seed)
    latent_dim = intrinsic_dim or min(10, dim, num_clusters + 2)

    basis_raw = rng.normal(size=(dim, latent_dim))
    basis, _ = np.linalg.qr(basis_raw)

    centers_latent = cluster_scale * rng.normal(size=(num_clusters, latent_dim))
    centers = centers_latent @ basis.T

    labels = rng.integers(0, num_clusters, size=n_samples)
    x = centers[labels] + noise * rng.normal(size=(n_samples, dim))

    return x
