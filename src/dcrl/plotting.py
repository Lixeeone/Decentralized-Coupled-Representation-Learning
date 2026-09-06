"""Publication-style figures drawn only from measured run artifacts."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#157F88", "#BD6A3C", "#5766AC", "#242F42"]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#253247", "text.color": "#253247",
        "axes.edgecolor": "#BBC3CD", "xtick.color": "#617087", "ytick.color": "#617087",
        "grid.alpha": .22, "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, out, name):
    fig.savefig(Path(out) / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(Path(out) / f"{name}.svg", bbox_inches="tight", metadata={"Date": None, "Creator": "DCRL"})
    plt.close(fig)


def plot_run(output):
    out = Path(output)
    style()
    with (out / "history.csv").open() as f:
        rows = list(csv.DictReader(f))
    step = np.array([int(r["step"]) for r in rows])
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), constrained_layout=True)
    for ax, key, title in zip(axes, ("E_orth", "E_sub", "E_off_can", "effective_rank"),
                              ("Row orthogonality", "Subspace alignment", "Canonical decorrelation", "Effective rank")):
        values = np.array([float(r[key]) if r.get(key) else np.nan for r in rows])
        ax.plot(step, values, color=COLORS[0], lw=2)
        if key != "effective_rank" and np.all(values[np.isfinite(values)] > 0):
            ax.set_yscale("log")
        ax.set(title=title, xlabel="Step", ylabel=key)
        ax.grid(True)
    fig.suptitle(f"Measured trajectory · target: {rows[0]['target']}", fontsize=12)
    save(fig, out, "invariants")
    spec = np.array([json.loads(r["latent_spectrum"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    for i in range(spec.shape[1]):
        ax.plot(step, spec[:, i], lw=1.5, alpha=.9, label=f"Mode {i + 1}")
    ax.set(xlabel="Step", ylabel="Normalized latent eigenvalue", title="Latent spectrum evolution")
    if spec.shape[1] <= 8:
        ax.legend(frameon=False)
    ax.grid(True)
    save(fig, out, "spectrum")


def plot_graph(rows, out):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for j, name in enumerate(dict.fromkeys(r["schedule"] for r in rows)):
        group = [r for r in rows if r["schedule"] == name]
        sizes = sorted(set(r["N"] for r in group))
        means, stds, stats = [], [], []
        for n in sizes:
            subset = [r for r in group if r["N"] == n and r["E_L_grid"] is not None]
            vals = [r["E_L_grid"] for r in subset]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
            stats.append(subset[0]["scaling_statistic"] if subset else np.nan)
        axes[0].errorbar(sizes, means, yerr=stds, marker="o", capsize=3, color=COLORS[j % 4], label=name)
        axes[1].plot(stats, means, "o-", color=COLORS[j % 4], label=name)
    for ax in axes:
        ax.set(xscale="log", yscale="log", ylabel="Finite-grid generator error")
        ax.grid(True)
        ax.legend(frameon=False)
    axes[0].set(xlabel="Substrate vertices N", title="Graph-to-diffusion diagnostic")
    axes[1].set(xlabel=r"$N\epsilon_N^{d+2}/\log N$", title="Scaling statistic")
    save(fig, out, "generator")


def plot_lyapunov(rows, residuals, out):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for ax, key, label in zip(axes[:2], ("V_orth", "E_sub"), ("Orthogonality Lyapunov function", "Principal-subspace error")):
        ax.semilogy([r["time"] for r in rows], [max(r[key], 1e-30) for r in rows], color=COLORS[0])
        ax.set(xlabel="Macroscopic time", ylabel=key, title=label)
        ax.grid(True)
    axes[2].loglog([r["eta_w"] for r in residuals], [r["absolute_residual"] for r in residuals], "o-", color=COLORS[1])
    axes[2].set(xlabel="Euler step size", ylabel="Absolute first-order remainder", title="Finite-step consistency")
    axes[2].grid(True)
    save(fig, out, "lyapunov")
