"""Command line entry point; every experiment writes its resolved protocol."""
from __future__ import annotations
import argparse
import json
from threadpoolctl import threadpool_limits
from .io import read_yaml, clean_json


def parser():
    p = argparse.ArgumentParser(prog="dcrl", description="DCRL numerical reproduction toolkit")
    p.add_argument("--threads", type=int, default=1, help="BLAS thread limit (default: 1)")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run synchronous or asynchronous coupled dynamics")
    run.add_argument("--config", required=True)
    run.add_argument("--output")
    run.add_argument("--steps", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--resume", help="checkpoint.npz; --steps is the absolute target horizon")
    for name in ("suite", "graph", "baselines"):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--output")
    l = sub.add_parser("lyapunov")
    l.add_argument("--output", default="results/lyapunov")
    b = sub.add_parser("benchmark")
    b.add_argument("--output", default="results/benchmark")
    b.add_argument("--samples", type=int, default=2048)
    b.add_argument("--dim", type=int, default=512)
    b.add_argument("--rank", type=int, default=64)
    b.add_argument("--repeats", type=int, default=10)
    plot = sub.add_parser("plot")
    plot.add_argument("--input", required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.threads < 1:
        p.error("--threads must be positive")
    try:
        with threadpool_limits(limits=args.threads):
            if args.command == "run":
                from .runner import run_experiment
                cfg = read_yaml(args.config)
                for k in ("output", "steps", "seed"):
                    if getattr(args, k) is not None:
                        cfg[k] = getattr(args, k)
                result = run_experiment(cfg, resume=args.resume)
                if result["status"] == "diverged":
                    raise SystemExit(2)
            elif args.command == "suite":
                from .runner import run_suite
                run_suite(args.config, args.output)
            elif args.command == "graph":
                from .graph import run_graph
                run_graph(args.config, args.output)
            elif args.command == "baselines":
                from .diagnostics import run_fixed_baselines
                run_fixed_baselines(args.config, args.output)
            elif args.command == "lyapunov":
                from .diagnostics import run_lyapunov
                run_lyapunov(args.output)
            elif args.command == "benchmark":
                from .diagnostics import run_benchmark
                run_benchmark(args.output, args.samples, args.dim, args.rank, args.repeats)
            elif args.command == "plot":
                from .plotting import plot_run
                plot_run(args.input)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        p.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
