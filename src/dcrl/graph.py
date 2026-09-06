"""Empirical Gibbs graph generators and analytic Langevin test functions.

No curve is synthesized from a desired convergence rate. Every reported error
is obtained by applying the two operators to the same fixed evaluation grid.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .io import read_yaml, write_csv, write_json, environment


def graph_generator(vertices, queries, values, query_values, potential,
                    query_potential, radius, diffusion=1., beta=1., manifold="circle"):
    vertices, queries = np.asarray(vertices), np.asarray(queries)
    values, query_values = np.asarray(values), np.asarray(query_values)
    if radius <= 0 or radius >= np.pi or diffusion <= 0 or beta < 0:
        raise ValueError("require 0 < radius < pi, D > 0, beta >= 0")
    if manifold not in {"circle", "sphere"}:
        raise ValueError("graph manifold must be circle or sphere")
    if values.ndim != 2 or query_values.shape != (len(queries), values.shape[1]):
        raise ValueError("function arrays must have shapes (N,p) and (Q,p)")
    if len(vertices) != len(values) or not np.isfinite(values).all():
        raise ValueError("invalid substrate function values")
    d = 1 if manifold == "circle" else 2
    if manifold == "circle":
        v = np.column_stack([np.cos(vertices), np.sin(vertices)])
        q = np.column_stack([np.cos(queries), np.sin(queries)])
    else:
        v, q = vertices, queries
    tree = cKDTree(v)
    neighborhoods = tree.query_ball_point(q, 2 * np.sin(radius / 2))
    result = np.full(query_values.shape, np.nan)
    degrees = np.zeros(len(queries), dtype=int)
    rate = 2 * diffusion * (d + 2) / radius ** 2
    for i, ids in enumerate(neighborhoods):
        ids = np.asarray(ids, dtype=int)
        # Exclude self transitions when a query is also a substrate point.
        ids = ids[np.linalg.norm(v[ids] - q[i], axis=1) > 1e-13]
        degrees[i] = len(ids)
        if not len(ids):
            continue
        logits = -0.5 * beta * (np.asarray(potential)[ids] - query_potential[i])
        weights = np.exp(logits - np.max(logits))
        weights /= weights.sum()
        result[i] = rate * (weights @ (values[ids] - query_values[i]))
    return result, degrees


def circle_functions(theta, diffusion, beta, amplitude):
    t = np.asarray(theta)
    f = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)])
    df = np.column_stack([np.cos(t), -np.sin(t), 2 * np.cos(2 * t), -2 * np.sin(2 * t)])
    ddf = -f * np.array([1, 1, 4, 4])
    potential = amplitude * np.cos(t)
    du = -amplitude * np.sin(t)
    return f, potential, diffusion * (ddf - beta * du[:, None] * df)


def sphere_functions(x, diffusion, beta, amplitude):
    x = np.asarray(x)
    f = np.column_stack([x[:, 0], x[:, 2], x[:, 0] ** 2, x[:, 2] ** 2])
    lap = np.column_stack([-2 * x[:, 0], -2 * x[:, 2], 2 - 6 * x[:, 0] ** 2, 2 - 6 * x[:, 2] ** 2])
    # U = amplitude*z, grad(U) = amplitude*(e_z - z*x).
    grad_u_x = -amplitude * x[:, 2] * x[:, 0]
    grad_u_z = amplitude * (1 - x[:, 2] ** 2)
    drift_dot = np.column_stack([grad_u_x, grad_u_z, 2 * x[:, 0] * grad_u_x, 2 * x[:, 2] * grad_u_z])
    return f, amplitude * x[:, 2], diffusion * (lap - beta * drift_dot)


def evaluation_grid(manifold, count):
    if manifold == "circle":
        return np.linspace(0, 2 * np.pi, count, endpoint=False)
    i = np.arange(count)
    z = 1 - 2 * (i + 0.5) / count
    phi = np.pi * (3 - np.sqrt(5)) * i
    return np.column_stack([np.sqrt(1 - z*z) * np.cos(phi), np.sqrt(1 - z*z) * np.sin(phi), z])


def run_graph(path, output=None):
    cfg = read_yaml(path)
    allowed = {"output", "manifold", "sizes", "seeds", "grid_size", "diffusion", "beta", "amplitude", "schedules"}
    if set(cfg) - allowed:
        raise ValueError("unknown graph configuration keys")
    out = Path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("graph output must be empty")
    manifold = cfg.get("manifold", "circle")
    if manifold not in {"circle", "sphere"}:
        raise ValueError("unknown manifold")
    D, beta, amplitude = cfg.get("diffusion", 1.), cfg.get("beta", 1.), cfg.get("amplitude", 0.5)
    functions = circle_functions if manifold == "circle" else sphere_functions
    grid = evaluation_grid(manifold, cfg.get("grid_size", 128))
    fq, uq, exact = functions(grid, D, beta, amplitude)
    rows = []
    for n in cfg["sizes"]:
        for seed in cfg["seeds"]:
            rng = np.random.default_rng(seed)
            if manifold == "circle":
                vertices = rng.uniform(0, 2 * np.pi, n)
            else:
                vertices = rng.normal(size=(n, 3))
                vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
            fv, uv, _ = functions(vertices, D, beta, amplitude)
            for schedule in cfg["schedules"]:
                radius = schedule["factor"] * (np.log(n) / n) ** schedule["exponent"]
                empirical, degrees = graph_generator(vertices, grid, fv, fq, uv, uq, radius, D, beta, manifold)
                complete = bool(np.all(degrees > 0))
                error = empirical - exact
                rows.append({"manifold": manifold, "N": n, "seed": seed, "schedule": schedule["name"],
                             "radius": radius, "scaling_statistic": n * radius ** ((1 if manifold == "circle" else 2) + 2) / np.log(n),
                             "E_L_grid": float(np.max(np.abs(error))) if complete else None,
                             "RMS_grid": float(np.sqrt(np.mean(error ** 2))) if complete else None,
                             "coverage": float(np.mean(degrees > 0)), "min_degree": int(degrees.min()),
                             "mean_degree": float(degrees.mean()), "grid_size": len(grid), "test_functions": 4})
    write_csv(out / "generator.csv", rows)
    write_json(out / "metadata.json", {"config": cfg, "environment": environment(),
               "definition": "maximum over the fixed finite evaluation grid and four named smooth test functions; not an operator norm"})
    from .plotting import plot_graph
    plot_graph(rows, out)
    print(f"Graph diagnostic: {len(rows)} measured rows -> {out}", flush=True)
    return rows
