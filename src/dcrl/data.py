"""Deterministic substrate generation and explicit feature preprocessing."""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
from .metrics import matrix


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def standardize(x):
    x = matrix(x).copy()
    center = x.mean(axis=0)
    x -= center
    scale = max(float(np.std(x)), 1e-12)
    return x / scale, center, scale


def generate(kind, n_samples=512, dim=32, seed=0, noise=0.1,
             intrinsic_dim=2, num_clusters=5, cluster_scale=4.0, density=0.05):
    if n_samples < 2 or dim < 1 or noise < 0:
        raise ValueError("invalid sample count, dimension, or noise")
    rng = np.random.default_rng(seed)
    if kind == "sphere":
        x = rng.normal(size=(n_samples, 3))
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    if kind == "torus":
        theta = []
        while len(theta) < n_samples:
            t = rng.uniform(0, 2 * np.pi, n_samples)
            theta.extend(t[rng.random(n_samples) < (2 + 0.7 * np.cos(t)) / 2.7].tolist())
        t = np.asarray(theta[:n_samples])
        p = rng.uniform(0, 2 * np.pi, n_samples)
        return np.column_stack([(2 + 0.7 * np.cos(t)) * np.cos(p),
                                (2 + 0.7 * np.cos(t)) * np.sin(p), 0.7 * np.sin(t)])
    if kind == "swiss_roll":
        t = 1.5 * np.pi * (1 + 2 * rng.random(n_samples))
        return np.column_stack([t * np.cos(t), rng.uniform(-6, 6, n_samples), t * np.sin(t)])
    if kind == "s_curve":
        t = rng.uniform(-1.5 * np.pi, 1.5 * np.pi, n_samples)
        return np.column_stack([np.sin(t), rng.uniform(-1, 1, n_samples), np.sign(t) * (np.cos(t) - 1)])
    if kind in {"cylinder", "hyperboloid"}:
        t, z = rng.uniform(0, 2 * np.pi, n_samples), rng.uniform(-1, 1, n_samples)
        r = np.sqrt(1 + z * z) if kind == "hyperboloid" else np.ones(n_samples)
        return np.column_stack([r * np.cos(t), r * np.sin(t), z])
    if kind == "intersecting_planes":
        x = rng.uniform(-1, 1, (n_samples, 3))
        plane = rng.integers(0, 2, n_samples)
        x[plane == 0, 2] = 0
        x[plane == 1, 1] = 0
        return x
    if not 1 <= intrinsic_dim <= dim or num_clusters < 2 or cluster_scale <= 0:
        raise ValueError("invalid latent mixture settings")
    if kind == "hd_gmm":
        basis, _ = np.linalg.qr(rng.normal(size=(dim, intrinsic_dim)))
        centers = cluster_scale * rng.normal(size=(num_clusters, intrinsic_dim)) @ basis.T
        return centers[rng.integers(0, num_clusters, n_samples)] + noise * rng.normal(size=(n_samples, dim))
    if kind == "sparse_hd":
        if not 0 < density <= 1:
            raise ValueError("density must be in (0,1]")
        x = rng.normal(size=(n_samples, dim)) * (rng.random((n_samples, dim)) < density)
        x[:, :intrinsic_dim] *= cluster_scale
        return x + noise * rng.normal(size=x.shape)
    raise ValueError(f"unknown substrate: {kind}")


def load_data(spec, seed):
    spec = dict(spec)
    preprocessing = spec.pop("preprocess", "none")
    kind = spec.pop("kind", "hd_gmm")
    info = {"kind": kind, "preprocess": preprocessing}
    if kind == "features":
        path = Path(spec.pop("path"))
        key = spec.pop("key", "features")
        max_samples = spec.pop("max_samples", None)
        expected = spec.pop("sha256", None)
        expected_dim = spec.pop("expected_dim", None)
        expected_samples = spec.pop("expected_samples", None)
        if spec:
            raise ValueError(f"unknown feature options: {sorted(spec)}")
        if not path.is_file():
            raise FileNotFoundError("Feature matrix missing. Supply the .npy/.npz path; see docs/DATA.md.")
        digest = sha256_file(path)
        if expected and digest != expected:
            raise ValueError("feature file SHA-256 does not match the configured value")
        if path.suffix == ".npy":
            x = np.load(path, mmap_mode="r", allow_pickle=False)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as f:
                x = f[key].copy()
        else:
            raise ValueError("features must be .npy or .npz, with no pickle objects")
        if x.ndim != 2 or x.dtype.kind not in "fi":
            raise ValueError("features must be a numeric matrix of shape (N,n)")
        if expected_dim is not None and x.shape[1] != expected_dim:
            raise ValueError("feature ambient dimension differs from expected_dim")
        if expected_samples is not None and len(x) != expected_samples:
            raise ValueError("feature sample count differs from expected_samples")
        if max_samples is not None:
            if max_samples < 2:
                raise ValueError("max_samples must be >= 2")
            indices = np.sort(np.random.default_rng(seed).choice(len(x), min(max_samples, len(x)), replace=False))
            x = x[indices]
            info["subset_indices_sha256"] = hashlib.sha256(indices.tobytes()).hexdigest()
        info.update({"file_sha256": digest, "file_name": path.name})
    else:
        x = generate(kind=kind, seed=seed, **spec)
    x = matrix(x).copy()
    center, scale = np.zeros(x.shape[1]), 1.0
    if preprocessing == "standardize":
        x, center, scale = standardize(x)
    elif preprocessing == "center":
        center = x.mean(axis=0)
        x -= center
    elif preprocessing != "none":
        raise ValueError("preprocess must be none, center, or standardize")
    info.update({"shape": list(x.shape), "center": center.tolist(), "scale": scale,
                 "prepared_sha256": hashlib.sha256(x.tobytes()).hexdigest()})
    return x, info
