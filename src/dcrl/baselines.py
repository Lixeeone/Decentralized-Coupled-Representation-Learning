"""Explicit fixed-stream reference algorithms; never substituted for DCRL."""
from __future__ import annotations
import numpy as np
from .dynamics import oja_delta
from .metrics import initialize_w, orthonormal_rows, principal_subspace, covariance_action


METHODS = ("exact_pca", "streaming_oja", "sanger", "eigengame", "distributed_orthogonal_iteration")


def run_baseline(x, rank, method, steps=2000, learning_rate=1e-3,
                 batch_size=64, seed=0, normalize_every=10, workers=4):
    if method not in METHODS or steps < 1 or batch_size < 1 or learning_rate <= 0 or normalize_every < 1:
        raise ValueError("invalid baseline configuration")
    if method == "exact_pca":
        return principal_subspace(x, rank, solver="dense")[0].T
    rng = np.random.default_rng(seed)
    w = initialize_w(rank, x.shape[1], seed + 17)
    # Deterministic shared stream schedule for the stochastic references.
    permutation, pos = rng.permutation(len(x)), 0
    shards = np.array_split(x, min(workers, len(x))) if workers > 0 else []
    if not shards:
        raise ValueError("workers must be positive")
    for step in range(steps):
        if pos >= len(x):
            permutation, pos = rng.permutation(len(x)), 0
        batch = x[permutation[pos:min(pos + batch_size, len(x))]]
        pos += len(batch)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if method == "streaming_oja":
                w += learning_rate * oja_delta(batch, w)
            elif method == "sanger":
                w += learning_rate * oja_delta(batch, w, "sanger")
            elif method == "eigengame":
                # Alpha-EigenGame utility gradient. Parent penalties use the
                # same minibatch covariance, divided by each parent's energy.
                cw = (batch @ w.T).T @ batch / len(batch)
                gram = w @ cw.T
                denominator = np.maximum(np.diag(gram), 1e-12)
                coefficients = np.tril(gram / denominator[None, :], k=-1)
                g = 2 * (cw - coefficients @ cw)
                g -= np.sum(g * w, axis=1, keepdims=True) * w
                w += learning_rate * g
                w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-15)
            else:
                # Synchronous all-reduce of local covariance actions, followed
                # by QR. A transparent distributed PCA reference, not DeEPCA.
                actions = [shard.T @ (shard @ w.T) for shard in shards]
                w = orthonormal_rows((sum(actions) / len(x)).T)
            if method in {"streaming_oja", "sanger"} and (step + 1) % normalize_every == 0:
                w = orthonormal_rows(w)
            if not np.isfinite(w).all():
                raise FloatingPointError("baseline diverged")
    # Postprocessing is part of the fixed-stream calibration and is disclosed.
    return orthonormal_rows(w)
