#!/usr/bin/env python3
"""Runs a training shell script over an alpha x beta grid, each cell
repeated across multiple seeds, and summarizes the results per cell.

Sibling of run_multi_seed.py (same core idea: several seeds per config to
tell a real effect from run-to-run training noise -- see that file's
docstring for why), extended to sweep two hyperparameters instead of just
comparing a couple of named configs. Given a 4x5 grid x 3 seeds is 60 full
training runs (each ~1-1.5h for libri_light_1hr), this is a multi-day,
almost certainly-gets-interrupted-at-some-point job, so cells that already
have a complete log (a prior run that produced both an eval_wer and a
train_runtime) are SKIPPED by default and their cached result reused --
pass --force to ignore existing logs and rerun everything.

Not hardcoded to any one fine-tuning set: --profile is forwarded as the
4th positional arg to `script` (see run_train_dynamic_grid.sh), so the
exact same grid tooling works for libri_light_1hr, libri_light_10hr,
libri_speech_clean_100hr, or a future gigaspeech_xs profile once that
dataset is prepared -- just point --profile (and probably --log-dir, to
keep results separate) at the new one, nothing else needs to change here.

Usage:
    python run_train_grid_seed.py run_train_dynamic_grid.sh \\
        --alphas 0.01 0.02 0.03 0.04 \\
        --betas 0.0 0.25 0.5 0.75 1.0 \\
        --seeds 0 1 2 \\
        --profile libri_light_1hr --log-dir grid_logs_1hr

    # Same grid, different fine-tuning set -- only --profile/--log-dir
    # change:
    python run_train_grid_seed.py run_train_dynamic_grid.sh \\
        --alphas 0.01 0.02 0.03 0.04 --betas 0.0 0.25 0.5 0.75 1.0 \\
        --profile libri_light_10hr --log-dir grid_logs_10hr

    # Resume an interrupted sweep (default behavior, no flag needed):
    python run_train_grid_seed.py run_train_dynamic_grid.sh \\
        --alphas 0.01 0.02 0.03 0.04 --betas 0.0 0.25 0.5 0.75 1.0 \\
        --profile libri_light_1hr --log-dir grid_logs_1hr

    # Split the grid across two GPUs by hand, e.g. two terminals:
    #   terminal 1: --alphas 0.01 0.02   (source set_config.sh, DEVICE_ID=0)
    #   terminal 2: --alphas 0.03 0.04   (export DEVICE_ID=1)
"""

import argparse
import ast
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "script",
        help="Training shell script accepting `bash <script> <alpha> "
             "<beta> <seed>` (see run_train_dynamic_grid.sh).")
    parser.add_argument(
        "--alphas", type=float, nargs="+", required=True,
        help="Alpha values to sweep.")
    parser.add_argument(
        "--betas", type=float, nargs="+", required=True,
        help="Beta values to sweep.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Seeds per (alpha, beta) cell (default: 0 1 2).")
    parser.add_argument(
        "--profile", type=str, default="libri_light_1hr",
        help="Fine-tuning set: forwarded as `script`'s 4th positional arg "
             "(any --finetune_profile choice in "
             "wav2vec2_finetuning_sets.py -- libri_light_1hr, "
             "libri_light_10hr, libri_speech_clean_100hr, or a future "
             "gigaspeech_xs profile once prepared). Default: "
             "libri_light_1hr.")
    parser.add_argument(
        "--log-dir", type=str, default="grid_logs",
        help="Directory for per-run raw logs and the combined summary. "
             "Use a different one per --profile (default: grid_logs) -- "
             "log filenames are already tagged with --profile so reusing "
             "one directory across profiles won't corrupt results, but "
             "separate directories make browsing/cleanup easier.")
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Rerun every cell even if a complete log already exists for "
             "it (default: skip and reuse already-completed cells, since "
             "a full sweep is a multi-day job and will likely get "
             "interrupted at some point).")
    return parser.parse_args()


def _fmt(x: float) -> str:
    """Matches wav2vec2_finetuning_sets.py's _fmt_float, so log file names
    line up with what a human would expect from the run_name (not
    strictly required, just for readability)."""
    return str(x).replace(".", "p").replace("-", "neg")


def try_load_cached(log_path: Path) -> Optional[Dict[str, Any]]:
    """If `log_path` exists and its content parses out both a final
    eval_metrics dict (with eval_wer) and a train_summary dict (with
    train_runtime) -- i.e. the run completed successfully last time --
    returns that as a result dict. Otherwise returns None (missing,
    partial, or crashed run -- needs to actually run/rerun).
    """
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


def run_one(script: str, alpha: float, beta: float, seed: int, profile: str,
           log_path: Path, tag: str) -> Dict[str, Any]:
    """Runs `bash <script> <alpha> <beta> <seed> <profile>`, streaming
    output live (prefixed with `tag`) while capturing it to `log_path` and
    scanning for the final HF Trainer metrics dicts. See
    run_multi_seed.py's run_one for the PYTHONUNBUFFERED rationale (same
    fix, same reason).
    """
    print(f"\n{'='*70}\n>>> {tag}\n{'='*70}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        ["bash", script, str(alpha), str(beta), str(seed), profile],
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


def print_grid(profile: str, alphas: List[float], betas: List[float],
               cell_summaries: Dict[Tuple[float, float], Dict[str, Any]],
               ) -> None:
    print(f"\n{'='*70}\neval_wer grid, {profile} "
          f"(mean ± std over seeds)\n{'='*70}")
    header = "alpha\\beta".ljust(10) + "".join(f"{b:>16.2f}" for b in betas)
    print(header)
    for a in alphas:
        row = f"{a:<10.3f}"
        for b in betas:
            s = cell_summaries.get((a, b), {})
            if "eval_wer" in s:
                m, sd = s["eval_wer"]["mean"], s["eval_wer"]["std"]
                row += f"{m:>9.4f}±{sd:<5.4f}"
            else:
                row += f"{'n/a':>16}"
        print(row)


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    cells = list(itertools.product(args.alphas, args.betas))
    total_runs = len(cells) * len(args.seeds)
    print(f"Grid ({args.profile}): {len(args.alphas)} alphas x "
          f"{len(args.betas)} betas = {len(cells)} cells x "
          f"{len(args.seeds)} seeds = {total_runs} runs total")

    # Load any existing summary.json (e.g. from a prior sweep over a
    # different --alphas/--betas subset against this same --log-dir) so
    # this run MERGES new cells into it instead of clobbering
    # previously-completed cells' results.
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    cell_summaries: Dict[Tuple[float, float], Dict[str, Any]] = {}
    summary_path = log_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            prev = json.load(f)
        all_results.update(prev.get("results", {}))
        for key, s in prev.get("summary", {}).items():
            rest = key.split("_alpha=", 1)[1]
            a_str, b_str = rest.split("_beta=", 1)
            cell_summaries[(float(a_str), float(b_str))] = s
    run_idx = 0
    n_skipped = 0

    for alpha, beta in cells:
        cell_key = f"profile={args.profile}_alpha={alpha}_beta={beta}"
        cell_results = []
        for seed in args.seeds:
            run_idx += 1
            log_path = (log_dir /
                       f"{args.profile}_alpha{_fmt(alpha)}_beta{_fmt(beta)}"
                       f"_seed{seed}.log")
            tag = (f"[{run_idx}/{total_runs} profile={args.profile} "
                  f"alpha={alpha} beta={beta} seed={seed}]")

            cached = None if args.force else try_load_cached(log_path)
            if cached is not None:
                n_skipped += 1
                print(f"{tag} SKIP (already completed, reusing "
                      f"{log_path})")
                result = cached
            else:
                result = run_one(args.script, alpha, beta, seed,
                                 args.profile, log_path, tag)
                if result["returncode"] != 0:
                    print(f"!!! {tag} exited with code "
                          f"{result['returncode']} -- see {log_path}")
            cell_results.append(result)

        cell_summary = summarize_cell(cell_results)
        cell_summaries[(alpha, beta)] = cell_summary
        wer = cell_summary.get("eval_wer", {})
        print(f"  -> alpha={alpha} beta={beta}: "
              f"eval_wer mean={wer.get('mean', float('nan')):.4f} "
              f"std={wer.get('std', float('nan')):.4f} "
              f"({cell_summary['n_ok']}/{cell_summary['n_total']} OK)")
        all_results[cell_key] = cell_results

        # Write incrementally so a crash mid-sweep doesn't lose completed
        # cells' results, and so print_grid() below reflects everything
        # done so far even on a partial/interrupted run.
        summary_path = log_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "profile": args.profile,
                "results": all_results,
                "summary": {
                    f"profile={args.profile}_alpha={a}_beta={b}": s
                    for (a, b), s in cell_summaries.items()
                },
            }, f, indent=2)

    print(f"\n{n_skipped}/{total_runs} runs were skipped (already-completed "
          f"cells reused).")
    print_grid(args.profile, args.alphas, args.betas, cell_summaries)
    print(f"\nFull results + summary written to {log_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
