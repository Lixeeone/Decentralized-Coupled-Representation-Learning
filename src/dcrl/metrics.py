"""Uncentered second-moment diagnostics; Section 5 and Appendix D.

Centering belongs to data preparation. Evaluation never recenters an evolving
configuration. The Procrustes gauge is fixed against an external eigenbasis.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


def matrix(x, name="x"):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite, nonempty real matrix")
    return x


def orthonormal_rows(w):
    """QR representative; reject rank collapse instead of inventing directions."""
    w = matrix(w, "w")
    if w.shape[0] > w.shape[1]:
        raise ValueError("rank exceeds ambient dimension")
    if np.linalg.matrix_rank(w) < w.shape[0]:
        raise ValueError("representation has deficient row rank")
    q, r = np.linalg.qr(w.T, mode="reduced")
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return (q * signs).T


def initialize_w(rank, dim, seed=0, scale=1.0):
    if not 1 <= rank <= dim or not np.isfinite(scale) or scale <= 0:
        raise ValueError("require 1 <= rank <= dim and positive finite scale")
    return scale * orthonormal_rows(np.random.default_rng(seed).normal(size=(rank, dim)))


def covariance_action(x, q, block_size=4096):
    """Apply X.T X / N without constructing an n by n covariance."""
    out = np.zeros_like(q, dtype=np.float64)
    for start in range(0, len(x), block_size):
        b = x[start:start + block_size]
        out += b.T @ (b @ q)
    return out / len(x)


def principal_subspace(x, rank, solver="auto", tol=1e-10):
    """Return (eigenbasis, eigenvalues, diagnostic info), descending order.

    The iterative solver uses deterministic initialization and fails loudly on
    nonconvergence. Residuals and the cutoff eigengap are always returned.
    """
    x = matrix(x)
    n = x.shape[1]
    if not 1 <= rank <= n:
        raise ValueError("rank must be between 1 and ambient dimension")
    if solver not in {"auto", "dense", "iterative"}:
        raise ValueError("unknown eigensolver")
    k = min(rank + 1, n)
    dense = solver == "dense" or k >= n or (solver == "auto" and n <= 256)
    if dense:
        cov = x.T @ x / len(x)
        vals, vecs = np.linalg.eigh(cov)
        vals, vecs = vals[::-1], vecs[:, ::-1]
    else:
        op = LinearOperator((n, n), matvec=lambda z: covariance_action(x, z),
                            matmat=lambda z: covariance_action(x, z), dtype=np.float64)
        v0 = np.random.default_rng(1729).normal(size=n)
        vals, vecs = eigsh(op, k=k, which="LA", tol=tol, v0=v0)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
    q = vecs[:, :rank].copy()
    for j in range(rank):
        if q[np.argmax(np.abs(q[:, j])), j] < 0:
            q[:, j] *= -1
    leading = vals[:rank]
    residual = np.linalg.norm(covariance_action(x, q) - q * leading, ord="fro")
    gap = float(vals[rank - 1] - vals[rank]) if rank < n else None
    return q, leading, {"solver": "dense" if dense else "iterative",
                        "eigen_residual": float(residual), "cutoff_eigengap": gap,
                        "gap_resolved": bool(gap is None or gap > 10 * tol * max(abs(vals[0]), 1))}


def row_orthogonality_error(w):
    w = matrix(w, "w")
    return float(np.linalg.norm(w @ w.T - np.eye(len(w)), "fro"))


def lyapunov(w):
    return 0.25 * row_orthogonality_error(w) ** 2


def subspace_alignment_error(w, q_target):
    """Largest principal-angle sine, with rank loss assigned maximal error.

    Residual singular values avoid sqrt(1-cos(theta)^2) cancellation near zero.
    """
    w, q = matrix(w, "w"), matrix(q_target, "q_target")
    m, n = w.shape
    if q.shape != (n, m) or not np.allclose(q.T @ q, np.eye(m), atol=1e-7):
        raise ValueError("q_target must have shape (n,m) with orthonormal columns")
    if np.linalg.matrix_rank(w) < m:
        return 1.0
    u = orthonormal_rows(w).T
    return float(np.clip(np.linalg.norm(u - q @ (q.T @ u), 2), 0, 1))


def canonical_offdiag_mass(x, w, q_target):
    x, w, q = matrix(x), matrix(w, "w"), matrix(q_target, "q_target")
    if x.shape[1] != w.shape[1] or q.shape != (w.shape[1], len(w)):
        raise ValueError("incompatible data, weight, and target shapes")
    if not np.allclose(q.T @ q, np.eye(len(w)), atol=1e-7):
        raise ValueError("q_target must have orthonormal columns")
    y = x @ w.T
    sigma_y = y.T @ y / len(x)
    a, s, bt = np.linalg.svd(w @ q, full_matrices=False)
    norm = np.linalg.norm(sigma_y, "fro")
    if norm <= np.finfo(float).tiny or s[-1] <= np.finfo(float).eps * len(w) * s[0]:
        return float("nan")  # undefined gauge or zero latent energy
    r = a @ bt
    rotated = r.T @ sigma_y @ r
    return float(np.linalg.norm(rotated - np.diag(np.diag(rotated)), "fro") / norm)


def compute_diagnostics(x, w, q_target):
    x, w = matrix(x), matrix(w, "w")
    y = x @ w.T
    sigma_y = y.T @ y / len(x)
    eigs = np.maximum(np.linalg.eigvalsh(sigma_y)[::-1], 0)
    energy = eigs.sum()
    p = eigs[eigs > 0] / energy if energy > 0 else np.array([])
    if np.linalg.matrix_rank(w) == len(w):
        q_w = orthonormal_rows(w)
        var_explained = np.sum((x @ q_w.T) ** 2) / max(np.sum(x ** 2), np.finfo(float).tiny)
    else:
        var_explained = float("nan")
    return {"E_orth": row_orthogonality_error(w),
            "E_sub": subspace_alignment_error(w, q_target),
            "E_off_can": canonical_offdiag_mass(x, w, q_target),
            "V_orth": lyapunov(w), "W_norm": float(np.linalg.norm(w)),
            "row_rank": int(np.linalg.matrix_rank(w)),
            "effective_rank": float(np.exp(-np.sum(p * np.log(p)))) if len(p) else 0.0,
            "variance_explained": float(var_explained),
            "latent_spectrum": (eigs / energy).tolist() if energy else eigs.tolist()}
