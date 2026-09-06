"""Averaged-flow, finite-step, fixed-stream, and timing experiments."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from scipy.integrate import solve_ivp
from .metrics import initialize_w, lyapunov, compute_diagnostics, principal_subspace
from .dynamics import oja_delta, DynamicsConfig, DCRL
from .geometry import make_domain
from .gossip import ADCRL
from .data import load_data
from .baselines import run_baseline, METHODS
from .io import read_yaml, write_csv, write_json, environment


def empty_output(path):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("output must be empty; choose a new directory")
    return out


def run_lyapunov(output):
    out = empty_output(output)
    spectrum = np.array([6., 3., 1., 0.4, 0.1, 0.05])
    sigma = np.diag(spectrum)
    w0 = initialize_w(2, 6, seed=7, scale=0.6)

    def drift(w):
        return (w * spectrum) - ((w * spectrum) @ w.T) @ w

    times = np.linspace(0, 10, 201)
    solution = solve_ivp(lambda t, z: drift(z.reshape(2, 6)).ravel(),
                         (0, 10), w0.ravel(), t_eval=times, rtol=1e-10, atol=1e-12)
    if not solution.success:
        raise RuntimeError(solution.message)
    rows = []
    # +/- eigenaxis points have exactly the requested second moment.
    x = np.concatenate([np.diag(np.sqrt(6 * spectrum)), -np.diag(np.sqrt(6 * spectrum))])
    q = np.eye(6)[:, :2]
    for t, z in zip(times, solution.y.T):
        w = z.reshape(2, 6)
        d = compute_diagnostics(x, w, q)
        a = w @ w.T - np.eye(2)
        derivative = -np.trace(a @ a @ w @ sigma @ w.T)
        rows.append({"time": float(t), "dV_dt": float(derivative), **{k: v for k, v in d.items() if k != "latent_spectrum"}})
    residuals = []
    f0 = drift(w0)
    a0 = w0 @ w0.T - np.eye(2)
    first = -np.trace(a0 @ a0 @ w0 @ sigma @ w0.T)
    for eta in np.logspace(-5, -1, 9):
        delta = lyapunov(w0 + eta * f0) - lyapunov(w0)
        residuals.append({"eta_w": float(eta), "delta_V": float(delta),
                          "first_order": float(eta * first), "absolute_residual": float(abs(delta - eta * first)),
                          "residual_over_eta_squared": float(abs(delta - eta * first) / eta ** 2)})
    write_csv(out / "averaged_flow.csv", rows)
    write_csv(out / "euler_residual.csv", residuals)
    write_json(out / "metadata.json", {"environment": environment(), "spectrum": spectrum,
                "seed": 7, "solver": "RK45", "rtol": 1e-10, "atol": 1e-12,
                "scope": "fixed-covariance numerical theorem diagnostic, reconstructed release configuration"})
    from .plotting import plot_lyapunov
    plot_lyapunov(rows, residuals, out)
    return rows


def run_fixed_baselines(path, output=None):
    cfg = read_yaml(path)
    if set(cfg) - {"output", "data", "seeds", "rank", "steps", "learning_rate", "batch_size", "normalize_every", "workers", "methods"}:
        raise ValueError("unknown baseline config keys")
    out = empty_output(output or cfg["output"])
    rows, info = [], []
    for seed in cfg.get("seeds", [0, 1, 2]):
        x, data_info = load_data(cfg["data"], seed)
        q, _, reference = principal_subspace(x, cfg["rank"])
        info.append({"seed": seed, "data": data_info, "target": reference})
        for method in cfg.get("methods", METHODS):
            status, metrics = "completed", {}
            begin = time.perf_counter()
            try:
                w = run_baseline(x, cfg["rank"], method, seed=seed,
                       **{k: cfg[k] for k in ("steps", "learning_rate", "batch_size", "normalize_every", "workers") if k in cfg})
                metrics = compute_diagnostics(x, w, q)
                np.savez_compressed(out / f"{method}-seed-{seed}.npz", w=w, q_target=q)
            except FloatingPointError:
                status = "diverged"
            rows.append({"method": method, "seed": seed, "status": status,
                         "elapsed_seconds": time.perf_counter() - begin,
                         **{k: v for k, v in metrics.items() if k != "latent_spectrum"}})
    write_csv(out / "baselines.csv", rows)
    write_json(out / "metadata.json", {"config": cfg, "environment": environment(), "inputs": info,
                 "normalization": "final QR for all iterative references; periodic QR for Oja/Sanger", 
                 "protocol_origin": "release reference settings; original Table D.4 schedules were not supplied"})
    return rows


def run_benchmark(output, n=2048, dim=512, rank=64, repeats=10):
    out = empty_output(output)
    if n < 2 or dim < 2 or not 1 <= rank <= dim or repeats < 2:
        raise ValueError("invalid benchmark dimensions or repeat count")
    x = np.random.default_rng(0).normal(size=(n, dim)).astype(np.float32)
    w = initialize_w(rank, dim, 0).astype(np.float32)
    # Timing only the specified primitive, with fresh input per call. Dense PCA
    # includes covariance construction. DCRL includes both synchronous phases.
    cfg = DynamicsConfig(rank=rank, init_scale=1.)
    domain = make_domain({"kind": "ball"}, x)
    center = np.asarray(domain.center, dtype=np.float32)
    def sync():
        rng = np.random.default_rng(3)
        drift = (x @ w.T) @ w
        next_x = x + np.float32(cfg.eta_x * cfg.diffusion) * drift + np.float32(np.sqrt(2 * cfg.diffusion * cfg.eta_x)) * rng.standard_normal(x.shape, dtype=np.float32)
        offset = next_x - center
        norm = np.linalg.norm(offset, axis=1, keepdims=True)
        next_x = center + offset * np.minimum(1, np.float32(domain.radius) / np.maximum(norm, np.float32(1e-12)))
        return w + np.float32(cfg.eta_w) * oja_delta(next_x, w)
    def gossip():
        xi, wi, wj = x[:1], w, w.copy()
        nx = xi + np.float32(cfg.eta_x * cfg.diffusion) * (xi @ wi.T) @ wi
        nx += np.float32(np.sqrt(2 * cfg.diffusion * cfg.eta_x)) * np.random.default_rng(3).standard_normal(nx.shape, dtype=np.float32)
        offset = nx - center
        nx = center + offset * np.minimum(1, np.float32(domain.radius) / np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), np.float32(1e-12)))
        wu = wi + np.float32(cfg.eta_w) * oja_delta(nx, wi)
        return .5 * (wu + wj), .5 * (wu + wj)
    tasks = [("Exact PCA", "one dense eigensolve", lambda: np.linalg.eigh(x.T @ x / n)),
             ("Streaming Oja", "one pass of N rank-one updates", lambda: _stream_pass(x, w, cfg.eta_w)),
             ("DCRL", "one synchronous spatial + plasticity step", sync),
             ("A-DCRL", "one active-edge primitive; no global state allocation", gossip)]
    rows = []
    for name, unit, call in tasks:
        call()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            call()
            times.append(1000 * (time.perf_counter() - start))
        rows.append({"method": name, "timing_unit": unit, "mean_ms": float(np.mean(times)),
                     "std_ms": float(np.std(times, ddof=1)), "repeats": repeats,
                     "parameter_bytes_per_copy": dim * dim * 4 if name == "Exact PCA" else rank * dim * 4,
                     "matrix_bytes_per_gossip_direction": rank * dim * 4 if name == "A-DCRL" else None})
    write_csv(out / "timings.csv", rows)
    env = environment()
    env["precision"] = "float32 timed arrays; platform BLAS/LAPACK"
    write_json(out / "metadata.json", {"environment": env, "N": n, "n": dim, "m": rank,
                  "seed": 0, "warmup": 1, "note": "different timing units; no end-to-end speedup claim"})
    return rows


def _stream_pass(x, w, eta):
    result = w.copy()
    for xi in x:
        result += eta * oja_delta(xi[None, :], result)
    return result
