"""Reproducible runs, explicit target protocols, divergence records, and resume."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from .data import load_data
from .dynamics import DCRL, DynamicsConfig
from .geometry import make_domain
from .gossip import ADCRL
from .metrics import compute_diagnostics, principal_subspace, lyapunov
from .io import (read_yaml, write_json, write_csv, deep_merge, fingerprint, source_fingerprint,
                 environment, save_checkpoint, load_checkpoint)


def resolve_config(cfg):
    defaults = {"name": "dcrl-run", "method": "dcrl", "seed": 0, "steps": 1000,
                "log_every": 100, "checkpoint_every": 1000,
                "output": "results/run", "target": "final", "plot": True,
                "data": {"kind": "hd_gmm", "n_samples": 512, "dim": 32, "intrinsic_dim": 2, "preprocess": "standardize"},
                "domain": {"kind": "ball"}, "dynamics": asdict(DynamicsConfig()),
                "gossip": {"alpha": 0.5, "graph": "ring"},
                "protocol_origin": "release reconstruction"}
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ValueError(f"unknown run keys: {sorted(unknown)}")
    out = deep_merge(defaults, cfg)
    # Data/domain mappings are complete specifications, not partial defaults.
    for key in ("data", "domain"):
        if key in cfg:
            out[key] = dict(cfg[key])
    DynamicsConfig(**out["dynamics"])
    if out["method"] not in {"dcrl", "adcrl"} or out["target"] not in {"initial", "final", "current"}:
        raise ValueError("invalid method or target protocol")
    for key in ("steps", "seed", "log_every", "checkpoint_every"):
        if not isinstance(out[key], int) or out[key] < (1 if key.endswith("every") else 0):
            raise ValueError(f"invalid {key}")
    if out["data"].get("preprocess", "none") != "none" and out["domain"].get("kind") not in {"ball", "identity"}:
        raise ValueError("analytic geometry requires preprocess: none")
    return out


def run_experiment(cfg, resume=None):
    cfg = resolve_config(cfg)
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if resume is None and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}; use a new directory or --resume")
    dynamic = DynamicsConfig(**cfg["dynamics"])
    semantic = {k: v for k, v in cfg.items() if k not in {"steps", "output", "name", "plot", "log_every", "checkpoint_every"}}
    digest = fingerprint(semantic)
    history = []
    deltas = {"count": 0, "sum": 0.0, "max_positive": 0.0, "violations": 0}
    if resume:
        arrays, meta = load_checkpoint(resume)
        if meta["config_sha256"] != digest:
            raise ValueError("resume configuration differs in algorithm, data, seed, or target protocol")
        if meta["source_sha256"] != source_fingerprint():
            raise ValueError("resume source differs from the checkpoint's implementation")
        x0, data_info = arrays["x0"], meta["data_info"]
    else:
        x0, data_info = load_data(cfg["data"], cfg["seed"])
    domain = make_domain(cfg["domain"], x0)
    state = (DCRL(x0, dynamic, domain, cfg["seed"]) if cfg["method"] == "dcrl"
             else ADCRL(x0, dynamic, domain, cfg["seed"], **cfg["gossip"]))
    q_initial, _, initial_info = principal_subspace(x0, dynamic.rank)
    if resume:
        if meta["step"] > cfg["steps"]:
            raise ValueError("steps is an absolute horizon and must be >= checkpoint step")
        state.x = arrays["x"]
        if cfg["method"] == "dcrl":
            state.w = arrays["w"]
        else:
            state.w_agents = arrays["w_agents"]
        state.rng.bit_generator.state = meta["rng_state"]
        state.step = meta["step"]
        history, deltas = meta["history"], meta["lyapunov_increments"]
    metadata = {"environment": environment(), "config_sha256": digest, "data": data_info,
                "domain": asdict(domain), "initial_target": initial_info,
                "trajectory_target": "current" if cfg["target"] == "current" else "initial",
                "endpoint_target": cfg["target"], "resumed_from_step": state.step if resume else None}
    public_config = deep_merge(cfg, {})
    public_config["output"] = "."
    if cfg["data"]["kind"] == "features":
        public_config["data"] = {**cfg["data"], "path": data_info["file_name"]}
    write_json(out / "config.json", public_config)
    write_json(out / "metadata.json", metadata)

    def log():
        if history and history[-1]["step"] == state.step:
            return
        q = principal_subspace(state.x, dynamic.rank)[0] if cfg["target"] == "current" else q_initial
        row = {"step": state.step, "target": metadata["trajectory_target"],
               **compute_diagnostics(state.x, state.w, q)}
        if cfg["method"] == "adcrl":
            row["consensus_error"] = state.consensus_error()
        history.append(row)

    def checkpoint():
        arrays = {"x0": x0, "x": state.x, "w": state.w}
        if cfg["method"] == "adcrl":
            arrays["w_agents"] = state.w_agents
        save_checkpoint(out / "checkpoint.npz", arrays,
                        {"step": state.step, "rng_state": state.rng.bit_generator.state,
                         "source_sha256": source_fingerprint(),
                         "config_sha256": digest, "data_info": data_info,
                         "history": history, "lyapunov_increments": deltas})

    log()
    status, error = "completed", None
    start = time.perf_counter()
    while state.step < cfg["steps"]:
        before = lyapunov(state.w) if cfg["method"] == "dcrl" else None
        try:
            state.advance()
        except FloatingPointError as exc:
            status, error = "diverged", str(exc)
            break
        if before is not None:
            delta = lyapunov(state.w) - before
            deltas["count"] += 1
            deltas["sum"] += delta
            deltas["max_positive"] = max(deltas["max_positive"], delta)
            deltas["violations"] += int(delta > 1e-12 * (1 + abs(before)))
        if state.step % cfg["log_every"] == 0:
            log()
            print(f"{cfg['name']}: step {state.step}/{cfg['steps']}", flush=True)
        if state.step % cfg["checkpoint_every"] == 0:
            checkpoint()
    elapsed = time.perf_counter() - start
    log()
    checkpoint()
    if cfg["target"] == "initial":
        q, target_info = q_initial, initial_info
    else:
        q, _, target_info = principal_subspace(state.x, dynamic.rank)
    endpoint = compute_diagnostics(state.x, state.w, q)
    summary = {"name": cfg["name"], "method": cfg["method"], "seed": cfg["seed"],
               "status": status, "error": error, "completed_steps": state.step,
               "requested_steps": cfg["steps"], "endpoint_target": cfg["target"],
               "metric_state": "final" if status == "completed" else "last_finite_before_divergence",
               "target_diagnostics": target_info, "elapsed_seconds_this_invocation": elapsed,
               **endpoint, "lyapunov_increments": deltas}
    if deltas["count"]:
        summary["lyapunov_increments"].update({"mean": deltas["sum"] / deltas["count"],
                    "violation_fraction": deltas["violations"] / deltas["count"],
                    "violation_threshold": "1e-12 * (1 + abs(V_before))"})
    if cfg["method"] == "adcrl":
        summary["consensus_error"] = state.consensus_error()
    write_csv(out / "history.csv", history)
    write_json(out / "summary.json", summary)
    np.savez_compressed(out / "final.npz", x=state.x, w=state.w, q_target=q,
                        **({"w_agents": state.w_agents} if cfg["method"] == "adcrl" else {}))
    if cfg["plot"]:
        from .plotting import plot_run
        plot_run(out)
    print(f"{cfg['name']}: {status}; E_sub={endpoint['E_sub']:.6g}; output={out}", flush=True)
    return summary


def run_suite(path, output=None):
    spec = read_yaml(path)
    if set(spec) - {"name", "output", "jobs", "seeds"}:
        raise ValueError("unknown suite configuration keys")
    out = Path(output or spec.get("output", "results/suite"))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("suite output is not empty; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    rows, collected = [], []
    for job in spec["jobs"]:
        if set(job) - {"config", "name", "overrides"}:
            raise ValueError("unknown job keys")
        base = read_yaml(Path(path).parent / job["config"])
        base = deep_merge(base, job.get("overrides", {}))
        for seed in spec.get("seeds", [0, 1, 2]):
            name = job["name"]
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("job names must be simple directory names")
            cfg = deep_merge(base, {"seed": seed, "name": name, "output": str(out / name / f"seed-{seed}")})
            summary = run_experiment(cfg)
            collected.append(summary)
            rows.append({k: v for k, v in summary.items() if not isinstance(v, (dict, list))})
            write_csv(out / "endpoints.csv", rows)
    aggregate = []
    for name in dict.fromkeys(s["name"] for s in collected):
        group = [s for s in collected if s["name"] == name]
        okay = [s for s in group if s["status"] == "completed"]
        row = {"name": name, "runs": len(group), "completed": len(okay), "diverged": len(group) - len(okay)}
        for key in ("E_orth", "E_sub", "E_off_can", "effective_rank", "variance_explained"):
            values = [s[key] for s in okay if s.get(key) is not None and np.isfinite(s[key])]
            row[f"{key}_n"] = len(values)
            row[f"{key}_mean"] = float(np.mean(values)) if values else None
            row[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregate.append(row)
    write_csv(out / "aggregate.csv", aggregate)
    write_json(out / "suite.json", spec)
    return aggregate
