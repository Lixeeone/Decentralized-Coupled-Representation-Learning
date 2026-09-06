#!/usr/bin/env python3
"""Reference diagnostics for the anonymous DCRL implementation.

This script runs compact reference diagnostics for:

1. synchronous DCRL;
2. asynchronous pairwise-gossip A-DCRL;
3. invariant metrics used in the paper:
   E_orth, E_sub, and E_off_can.

For synchronous DCRL, endpoint subspace diagnostics are computed against the
principal subspace of the final configuration. This matches the coupled setting,
where the spatial configuration may evolve during the reference run.

For A-DCRL, the script reports communication-oriented reference diagnostics:
row-orthogonality of the averaged local copy and consensus disagreement among
local matrices. This reflects the role of A-DCRL as a pairwise-gossip reference
implementation rather than a separate convergence theorem.

The script is intentionally lightweight. It uses self-contained reference inputs
only and does not attempt to reproduce every large-scale feature experiment in
the paper.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.adcrl_gossip import run_adcrl
from src.data import make_hd_gmm, make_swiss_roll, standardize
from src.dcrl_sync import run_dcrl
from src.metrics import (
    compute_diagnostics,
    row_orthogonality_error,
    top_principal_subspace,
)
from src.utils import set_random_seed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to a CSV file."""
    if not rows:
        raise ValueError("No rows were provided for CSV export.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_endpoint(
    *,
    method: str,
    substrate: str,
    x_final: np.ndarray,
    w_final: np.ndarray,
    q_target: np.ndarray,
    num_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Compute endpoint invariant diagnostics for one synchronous reference run."""
    diagnostics = compute_diagnostics(
        x=x_final,
        w=w_final,
        q_target=q_target,
    )

    return {
        "method": method,
        "substrate": substrate,
        "steps": num_steps,
        "seed": seed,
        "E_orth": diagnostics["E_orth"],
        "E_sub": diagnostics["E_sub"],
        "E_off_can": diagnostics["E_off_can"],
    }


def consensus_error(w_agents: np.ndarray, w_mean: np.ndarray) -> float:
    """Compute average Frobenius disagreement from the mean local copy."""
    if w_agents.ndim != 3:
        raise ValueError("w_agents must have shape (N, m, n).")
    if w_mean.ndim != 2:
        raise ValueError("w_mean must have shape (m, n).")

    centered = w_agents - w_mean[None, :, :]
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=(1, 2)))))


def summarize_gossip(
    *,
    method: str,
    substrate: str,
    w_agents: np.ndarray,
    w_mean: np.ndarray,
    num_rounds: int,
    seed: int,
) -> dict[str, Any]:
    """Compute compact diagnostics for one A-DCRL gossip reference run."""
    return {
        "method": method,
        "substrate": substrate,
        "rounds": num_rounds,
        "seed": seed,
        "E_orth_mean": row_orthogonality_error(w_mean),
        "consensus_error": consensus_error(w_agents=w_agents, w_mean=w_mean),
    }


def run_sync_reference(output_dir: Path, seed: int) -> None:
    """Run compact synchronous DCRL diagnostics on reference substrates."""
    rows: list[dict[str, Any]] = []

    # Low-dimensional reference substrate.
    swiss_steps = 5000
    x_swiss = standardize(make_swiss_roll(n_samples=512, seed=seed))

    sync_swiss = run_dcrl(
        x0=x_swiss,
        rank=2,
        num_steps=swiss_steps,
        eta_x=2.0e-3,
        eta_w=5.0e-4,
        diffusion=1.0e-2,
        beta=1.0,
        seed=seed,
        stabilize_every=None,
    )

    # The target subspace is computed from the final configuration because the
    # coupled run updates the spatial state.
    q_swiss = top_principal_subspace(sync_swiss["x"], rank=2)

    rows.append(
        summarize_endpoint(
            method="DCRL",
            substrate="swiss_roll",
            x_final=sync_swiss["x"],
            w_final=sync_swiss["w"],
            q_target=q_swiss,
            num_steps=swiss_steps,
            seed=seed,
        )
    )

    # Controlled high-dimensional reference substrate.
    hd_steps = 3000
    hd_rank = 2

    x_hd = standardize(
        make_hd_gmm(
            n_samples=512,
            dim=32,
            num_clusters=3,
            intrinsic_dim=hd_rank,
            noise=0.10,
            seed=seed + 1,
        )
    )

    sync_hd = run_dcrl(
        x0=x_hd,
        rank=hd_rank,
        num_steps=hd_steps,
        eta_x=1.0e-3,
        eta_w=1.0e-3,
        diffusion=1.0e-4,
        beta=1.0,
        seed=seed + 1,
        stabilize_every=None,
    )

    # As above, compare with the final empirical covariance of the evolved
    # configuration.
    q_hd = top_principal_subspace(sync_hd["x"], rank=hd_rank)

    rows.append(
        summarize_endpoint(
            method="DCRL",
            substrate="hd_gmm",
            x_final=sync_hd["x"],
            w_final=sync_hd["w"],
            q_target=q_hd,
            num_steps=hd_steps,
            seed=seed + 1,
        )
    )

    write_csv(output_dir / "sync_dcrl_diagnostics.csv", rows)


def run_async_reference(output_dir: Path, seed: int) -> None:
    """Run compact asynchronous A-DCRL pairwise-gossip diagnostics."""
    rows: list[dict[str, Any]] = []

    async_rounds = 5000

    x_hd = standardize(
        make_hd_gmm(
            n_samples=256,
            dim=32,
            num_clusters=3,
            intrinsic_dim=2,
            noise=0.10,
            seed=seed + 10,
        )
    )

    async_hd = run_adcrl(
        x0=x_hd,
        rank=2,
        num_rounds=async_rounds,
        eta_w=1.0e-3,
        gossip_alpha=0.5,
        graph_type="ring",
        seed=seed + 10,
    )

    rows.append(
        summarize_gossip(
            method="A-DCRL",
            substrate="hd_gmm_ring",
            w_agents=async_hd["w_agents"],
            w_mean=async_hd["w_mean"],
            num_rounds=async_rounds,
            seed=seed + 10,
        )
    )

    write_csv(output_dir / "adcrl_gossip_diagnostics.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compact reference diagnostics for DCRL and A-DCRL."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Directory for diagnostic CSV outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for reference diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_sync_reference(output_dir=args.output_dir, seed=args.seed)
    run_async_reference(output_dir=args.output_dir, seed=args.seed)

    print(f"Reference diagnostics written to: {args.output_dir}")


if __name__ == "__main__":
    main()
