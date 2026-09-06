"""Utility functions for the anonymous DCRL reference implementation."""

from __future__ import annotations

import numpy as np


def set_random_seed(seed: int) -> None:
    """Set NumPy's global seed for deterministic top-level scripts."""
    np.random.seed(seed)


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create a local NumPy random generator."""
    return np.random.default_rng(seed)


def row_orthonormalize(w: np.ndarray) -> np.ndarray:
    """Return a row-orthonormal representative from QR(W.T).

    For W in R^{m x n}, this returns a matrix with orthonormal rows spanning
    the same numerical row space when W has full row rank.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("w must have shape (m, n).")

    m, n = w.shape
    if m <= 0 or n <= 0:
        raise ValueError("w must be non-empty.")
    if m > n:
        raise ValueError("row_orthonormalize expects m <= n.")

    q, _ = np.linalg.qr(w.T, mode="reduced")
    return q[:, :m].T


def initialize_w(
    rank: int,
    dim: int,
    seed: int = 0,
    scale: float = 1.0,
) -> np.ndarray:
    """Initialize a full-row-rank representation matrix W in R^{m x n}."""
    if rank <= 0 or dim <= 0:
        raise ValueError("rank and dim must be positive.")
    if rank > dim:
        raise ValueError("rank must not exceed dim.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    rng = np.random.default_rng(seed)
    w = rng.normal(size=(rank, dim))
    return scale * row_orthonormalize(w)


def project_rows_to_ball(
    x: np.ndarray,
    radius: float = 5.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Project rows of X onto a Euclidean ball.

    This implements a simple substrate-domain retraction Pi_X for compact
    reference diagnostics.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must have shape (N, n).")
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    factors = np.minimum(1.0, radius / (norms + eps))
    return x * factors


def require_finite(name: str, array: np.ndarray) -> None:
    """Raise an error if an array contains non-finite values."""
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"{name} contains non-finite values.")
