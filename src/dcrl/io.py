"""Portable, pickle-free outputs and anonymous environment provenance."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import numpy as np
import yaml


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path, obj):
    Path(path).write_text(json.dumps(clean_json(obj), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in clean_json(row).items()})


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("configuration must be a YAML mapping")
    return cfg


def deep_merge(a, b):
    out = dict(a)
    for key, val in b.items():
        out[key] = deep_merge(out[key], val) if isinstance(out.get(key), dict) and isinstance(val, dict) else val
    return out


def fingerprint(obj):
    return hashlib.sha256(json.dumps(clean_json(obj), sort_keys=True).encode()).hexdigest()


def source_fingerprint():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def environment():
    # No username, hostname, absolute working directory, or general env dump.
    versions = {}
    for name in ("numpy", "scipy", "matplotlib", "PyYAML", "threadpoolctl", "dcrl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "source-tree"
    from threadpoolctl import threadpool_info
    blas = [{k: item.get(k) for k in ("internal_api", "version", "threading_layer", "architecture", "num_threads")}
            for item in threadpool_info()]
    return {"python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "packages": versions, "blas": blas,
            "precision": "float64", "source_sha256": source_fingerprint()}


def save_checkpoint(path, arrays, meta):
    """Atomic same-directory replacement; np.load never needs pickle."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays, metadata=np.array(json.dumps(clean_json(meta), allow_nan=False)))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files if k != "metadata"}, json.loads(str(f["metadata"]))
