"""Invariant diagnostics for the anonymous DCRL reference implementation."""

from __future__ import annotations

import numpy as np

from .utils import row_orthonormalize


def top_principal_subspace(x: np.ndarray, rank: int) -> np.ndarray:
    """Return the top-rank eigenvector matrix of the empirical covariance.

    The returned matrix Q has shape (n, rank), matching the paper's convention
    for the target principal eigenspace.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must have shape (N, n).")

    n_samples, dim = x.shape
    if n_samples <= 0:
        raise ValueError("x must contain at least one sample.")
    if rank <= 0 or rank > dim:
        raise ValueError("rank must satisfy 0 < rank <= n.")

    cov = (x.T @ x) / n_samples
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return vectors[:, order[:rank]]


def _validate_subspace_shapes(w: np.ndarray, q_target: np.ndarray) -> None:
    """Validate shape compatibility between Row(W) and Col(Q_target)."""
    if w.ndim != 2:
        raise ValueError("w must have shape (m, n).")
    if q_target.ndim != 2:
        raise ValueError("q_target must have shape (n, m).")

    m, dim = w.shape
    q_dim, q_rank = q_target.shape

    if q_dim != dim:
        raise ValueError("q_target must have the same ambient dimension as w.")
    if q_rank != m:
        raise ValueError("q_target must have the same rank as the number of rows in w.")


def row_orthogonality_error(w: np.ndarray) -> float:
    """Compute E_orth = ||W W^T - I_m||_F."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("w must have shape (m, n).")

    m = w.shape[0]
    return float(np.linalg.norm(w @ w.T - np.eye(m), ord="fro"))


def subspace_alignment_error(w: np.ndarray, q_target: np.ndarray) -> float:
    """Compute E_sub as the sine of the largest principal angle.

    This compares Row(W) with Col(Q_target).
    """
    w = np.asarray(w, dtype=np.float64)
    q_target = np.asarray(q_target, dtype=np.float64)
    _validate_subspace_shapes(w, q_target)

    q_w = row_orthonormalize(w).T
    singular_values = np.linalg.svd(q_w.T @ q_target, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)

    min_cosine = float(np.min(singular_values))
    return float(np.sqrt(max(0.0, 1.0 - min_cosine**2)))


def canonical_offdiag_mass(
    x: np.ndarray,
    w: np.ndarray,
    q_target: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute the canonical-gauge covariance off-diagonal mass.

    The gauge is fixed by the Procrustes alignment of W Q_target, independent
    of the measured latent covariance.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    q_target = np.asarray(q_target, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("x must have shape (N, n).")
    if x.shape[0] <= 0:
        raise ValueError("x must contain at least one sample.")
    if x.shape[1] != w.shape[1]:
        raise ValueError("x and w must have the same ambient dimension.")

    _validate_subspace_shapes(w, q_target)

    y = x @ w.T
    sigma_y = (y.T @ y) / x.shape[0]

    alignment = w @ q_target
    u, _, vt = np.linalg.svd(alignment, full_matrices=False)
    r = u @ vt

    rotated = r.T @ sigma_y @ r
    offdiag = rotated - np.diag(np.diag(rotated))

    denom = np.linalg.norm(sigma_y, ord="fro")
    return float(np.linalg.norm(offdiag, ord="fro") / max(denom, eps))


def compute_diagnostics(
    x: np.ndarray,
    w: np.ndarray,
    q_target: np.ndarray,
) -> dict[str, float]:
    """Compute the invariant diagnostics used by the reference script."""
    return {
        "E_orth": row_orthogonality_error(w),
        "E_sub": subspace_alignment_error(w, q_target),
        "E_off_can": canonical_offdiag_mass(x, w, q_target),
    }
