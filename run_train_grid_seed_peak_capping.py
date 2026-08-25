#!/usr/bin/env python3
"""Runs a training shell script over an alpha sweep, each value repeated
across multiple seeds, and summarizes the results.

1-D sibling of run_train_grid_seed.py (same resumable/multi-seed idea,
see that file's docstring), for the peak-capping SETS variant, which is
driven by a single --alpha (the confidence-cap parameter, cap = 1 -
alpha) instead of --alpha/--beta. Cells with an already-complete log are
skipped by default (pass --force to redo everything) -- see
run_train_grid_seed.py for why.

Usage:
    python run_train_grid_seed_peak_capping.py \\
        run_train_dynamic_grid_peak_capping.sh \\
        --alphas 0.05 0.10 0.15 0.20 \\
        --seeds 0 1 2 \\
        --profile libri_light_1hr --log-dir grid_logs_1hr_peak_capping
"""

import argparse
import ast
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "script",
        help="Training shell script accepting `bash <script> <alpha> "
             "<seed> <profile>` (see "
             "run_train_dynamic_grid_peak_capping.sh).")
    parser.add_argument(
        "--alphas", type=float, nargs="+", required=True,
        help="Alpha (confidence-cap) values to sweep.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Seeds per alpha value (default: 0 1 2).")
    parser.add_argument(
        "--profile", type=str, default="libri_light_1hr",
        help="Fine-tuning set: forwarded as `script`'s 3rd positional "
             "arg (default: libri_light_1hr).")
    parser.add_argument(
        "--log-dir", type=str, default="grid_logs_peak_capping",
        help="Directory for per-run raw logs and the combined summary.")
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Rerun every cell even if a complete log already exists.")
    return parser.parse_args()


def _fmt(x: float) -> str:
    """Matches wav2vec2_finetuning_sets.py's _fmt_float."""
    return str(x).replace(".", "p").replace("-", "neg")


def try_load_cached(log_path: Path) -> Optional[Dict[str, Any]]:
    """See run_train_grid_seed.py's try_load_cached -- identical logic."""
    if not log_path.exists():
        return None
    eval_metrics = None
    train_summary = None
    with open(log_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(parsed, dict):
                    continue
                if "eval_wer" in parsed:
                    eval_metrics = parsed
                elif "train_runtime" in parsed:
                    train_summary = parsed
    if eval_metrics is None or train_summary is None:
        return None
    return {
        "returncode": 0, "wall_time_s": None,
        "eval_metrics": eval_metrics, "train_summary": train_summary,
        "cached": True,
    }


def run_one(script: str, alpha: float, seed: int, profile: str,
           log_path: Path, tag: str) -> Dict[str, Any]:
    """Runs `bash <script> <alpha> <seed> <profile>`; see
    run_train_grid_seed.py's run_one for the PYTHONUNBUFFERED rationale."""
    print(f"\n{'='*70}\n>>> {tag}\n{'='*70}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        ["bash", script, str(alpha), str(seed), profile],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    eval_metrics: Optional[Dict[str, Any]] = None
    train_summary: Optional[Dict[str, Any]] = None

    with open(log_path, "w") as log_file:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(f"{tag} {line}")
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()

            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    continue
                if not isinstance(parsed, dict):
                    continue
                if "eval_wer" in parsed:
                    eval_metrics = parsed
                elif "train_runtime" in parsed:
                    train_summary = parsed

    returncode = proc.wait()
    wall_time_s = time.perf_counter() - t0

    return {
        "returncode": returncode, "wall_time_s": wall_time_s,
        "eval_metrics": eval_metrics, "train_summary": train_summary,
        "cached": False,
    }


def summarize_cell(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_runs = [
        r for r in results
        if r["returncode"] == 0 and r["eval_metrics"] is not None
    ]
    summary: Dict[str, Any] = {"n_total": len(results), "n_ok": len(ok_runs)}
    for key, source in (
        ("eval_wer", "eval_metrics"),
        ("eval_loss", "eval_metrics"),
        ("train_runtime", "train_summary"),
    ):
        values = [
            r[source][key] for r in ok_runs
            if r[source] is not None and key in r[source]
        ]
        if not values:
            continue
        summary[key] = {
            "values": values,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def print_sweep(profile: str, alphas: List[float],
                cell_summaries: Dict[float, Dict[str, Any]]) -> None:
    print(f"\n{'='*70}\neval_wer vs alpha (cap=1-alpha), {profile} "
          f"(mean ± std over seeds)\n{'='*70}")
    for a in alphas:
        s = cell_summaries.get(a, {})
        if "eval_wer" in s:
            m, sd = s["eval_wer"]["mean"], s["eval_wer"]["std"]
            print(f"  alpha={a:<6.3f} eval_wer={m:.4f} ± {sd:.4f}")
        else:
            print(f"  alpha={a:<6.3f} n/a")


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(args.alphas) * len(args.seeds)
    print(f"Sweep ({args.profile}): {len(args.alphas)} alphas x "
          f"{len(args.seeds)} seeds = {total_runs} runs total")

    # Load any existing summary.json (e.g. from a prior sweep over a
    # different set of --alphas against this same --log-dir) so this run
    # MERGES new cells into it instead of clobbering previously-completed
    # cells' results.
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    cell_summaries: Dict[float, Dict[str, Any]] = {}
    summary_path = log_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            prev = json.load(f)
        all_results.update(prev.get("results", {}))
        for key, s in prev.get("summary", {}).items():
            a = float(key.rsplit("alpha=", 1)[1])
            cell_summaries[a] = s
    run_idx = 0
    n_skipped = 0

    for alpha in args.alphas:
        cell_key = f"profile={args.profile}_alpha={alpha}"
        cell_results = []
        for seed in args.seeds:
            run_idx += 1
            log_path = (log_dir /
                       f"{args.profile}_alpha{_fmt(alpha)}_seed{seed}.log")
            tag = (f"[{run_idx}/{total_runs} profile={args.profile} "
                  f"alpha={alpha} seed={seed}]")

            cached = None if args.force else try_load_cached(log_path)
            if cached is not None:
                n_skipped += 1
                print(f"{tag} SKIP (already completed, reusing "
                      f"{log_path})")
                result = cached
            else:
                result = run_one(args.script, alpha, seed, args.profile,
                                 log_path, tag)
                if result["returncode"] != 0:
                    print(f"!!! {tag} exited with code "
                          f"{result['returncode']} -- see {log_path}")
            cell_results.append(result)

        cell_summary = summarize_cell(cell_results)
        cell_summaries[alpha] = cell_summary
        wer = cell_summary.get("eval_wer", {})
        print(f"  -> alpha={alpha}: "
              f"eval_wer mean={wer.get('mean', float('nan')):.4f} "
              f"std={wer.get('std', float('nan')):.4f} "
              f"({cell_summary['n_ok']}/{cell_summary['n_total']} OK)")
        all_results[cell_key] = cell_results

        summary_path = log_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "profile": args.profile,
                "results": all_results,
                "summary": {
                    f"profile={args.profile}_alpha={a}": s
                    for a, s in cell_summaries.items()
                },
            }, f, indent=2)

    print(f"\n{n_skipped}/{total_runs} runs were skipped (already-completed "
          f"cells reused).")
    print_sweep(args.profile, args.alphas, cell_summaries)
    print(f"\nFull results + summary written to {log_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
