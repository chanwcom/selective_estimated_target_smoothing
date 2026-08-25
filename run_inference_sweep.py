#!/usr/bin/env python3
"""Runs run_inference.sh (final WER via beam search, test-clean AND
test-other) over every checkpoint matching a glob pattern under a
checkpoints directory, and summarizes the results.

This is the "final performance" counterpart to run_train_grid_seed.py's
training-time eval_wer (which is only a fast, greedy-decoded, test-clean-
only proxy -- see the discussion that led to this script). Point it at the
same --checkpoint-top-dir the training runs wrote to, with a --pattern
matching the run-name convention from wav2vec2_finetuning_sets.py's
_default_run_name()/_batching_suffix() (e.g. "..._alpha_X_beta_Y_..._
seedN"), and it will:
  1. Find each matching run directory's highest-numbered checkpoint-N
     subdirectory (so it doesn't need to know --max_steps).
  2. Run `bash run_inference.sh <checkpoint_dir>` for it (both test splits
     in one call).
  3. If --name-regex extracts alpha/beta from the run directory name,
     group results into an alpha x beta grid (mean +/- std over seeds),
     printed once per test split. Otherwise falls back to a flat
     per-checkpoint listing.

Like run_train_grid_seed.py, this is a resumable, multi-hour-scale job: a
checkpoint whose log already has both splits' results is skipped by
default (pass --force to rerun everything).

Usage:
    python run_inference_sweep.py \\
        --pattern "libri_light_1hr_shc_2500steps_alpha_*_beta_*_unigram_32_dynbatch6400000_seed*"

    python run_inference_sweep.py \\
        --checkpoint-top-dir /mnt/data/home/chanwcom/models \\
        --pattern "libri_light_10hr_shc_5000steps_alpha_*_beta_*_unigram_32_bucket0_seed*" \\
        --log-dir inference_logs_10hr
"""

import argparse
import ast
import fnmatch
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_CHECKPOINT_TOP_DIR = "/mnt/data/home/chanwcom/models"
_DEFAULT_NAME_REGEX = r"alpha_(?P<alpha>[\w]+)_beta_(?P<beta>[\w]+)_unigram_\d+.*_seed(?P<seed>\d+)$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint-top-dir", type=str, default=_DEFAULT_CHECKPOINT_TOP_DIR,
        help=f"Parent directory holding one subdirectory per run (default: "
             f"{_DEFAULT_CHECKPOINT_TOP_DIR}, matches "
             f"--checkpoint_top_dir in the training scripts).")
    parser.add_argument(
        "--pattern", type=str, required=True,
        help="Glob pattern (fnmatch-style, e.g. with '*') matched against "
             "run directory basenames under --checkpoint-top-dir.")
    parser.add_argument(
        "--name-regex", type=str, default=_DEFAULT_NAME_REGEX,
        help="Regex with named groups 'alpha', 'beta', 'seed' applied to "
             "each matched run directory's basename, used to group "
             "results into an alpha x beta grid. If it doesn't match a "
             "given directory, that checkpoint's result is still "
             "reported, just outside the grid view.")
    parser.add_argument(
        "--inference-script", type=str, default="run_inference.sh",
        help="Script accepting `bash <script> <checkpoint_dir>` and "
             "printing one result dict per test split (default: "
             "run_inference.sh).")
    parser.add_argument(
        "--log-dir", type=str, default="inference_logs",
        help="Directory for per-checkpoint raw logs and the combined "
             "summary (default: inference_logs).")
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Rerun every checkpoint even if a complete log (both test "
             "splits) already exists for it.")
    return parser.parse_args()


def _unfmt(s: str) -> Any:
    """Best-effort reverse of _fmt_float (str(x).replace('.','p').replace
    ('-','neg')) used in the training scripts' run names, e.g. '0p01' ->
    0.01. Falls back to the raw string if it doesn't parse as a float --
    grid grouping/display just uses the string in that case."""
    try:
        return float(s.replace("neg", "-").replace("p", "."))
    except ValueError:
        return s


def find_checkpoints(top_dir: Path, pattern: str) -> List[Path]:
    """Finds each matching run directory's highest-numbered checkpoint-N
    subdirectory."""
    checkpoints = []
    if not top_dir.is_dir():
        return checkpoints
    for run_dir in sorted(top_dir.iterdir()):
        if not run_dir.is_dir() or not fnmatch.fnmatch(run_dir.name, pattern):
            continue
        step_dirs = [
            d for d in run_dir.iterdir()
            if d.is_dir() and re.match(r"^checkpoint-\d+$", d.name)
        ]
        if not step_dirs:
            print(f"!!! {run_dir} matched --pattern but has no "
                  f"checkpoint-N subdirectory -- skipping.")
            continue
        best = max(step_dirs, key=lambda d: int(d.name.split("-")[1]))
        checkpoints.append(best)
    return checkpoints


def try_load_cached(log_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Returns the parsed per-split result dicts if `log_path` already has
    a complete run (both test-clean and test-other present), else None."""
    if not log_path.exists():
        return None
    results = []
    with open(log_path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(parsed, dict) and "wer" in parsed and "test_split" in parsed:
                    results.append(parsed)
    splits_seen = {r["test_split"] for r in results}
    if {"test-clean", "test-other"} <= splits_seen:
        return results
    return None


def run_one(script: str, checkpoint_dir: Path, log_path: Path,
           tag: str) -> List[Dict[str, Any]]:
    """Runs `bash <script> <checkpoint_dir>`, streaming output live while
    capturing it and collecting EVERY {'wer': ..., 'test_split': ...}
    dict printed (run_inference.sh prints one per split, unlike training's
    single final eval dict)."""
    print(f"\n{'='*70}\n>>> {tag}\n{'='*70}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        ["bash", script, str(checkpoint_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    results: List[Dict[str, Any]] = []
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
                if isinstance(parsed, dict) and "wer" in parsed and "test_split" in parsed:
                    results.append(parsed)

    returncode = proc.wait()
    if returncode != 0:
        print(f"!!! {tag} exited with code {returncode} -- see {log_path}")
    return results


def summarize_grid(
    cell_results: Dict[Tuple[Any, Any], Dict[str, List[float]]],
) -> Dict[Tuple[Any, Any], Dict[str, Dict[str, float]]]:
    summary = {}
    for cell, per_split in cell_results.items():
        summary[cell] = {}
        for split, values in per_split.items():
            if not values:
                continue
            summary[cell][split] = {
                "values": values,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return summary


def print_grid(alphas: List[Any], betas: List[Any], split: str,
               summary: Dict[Tuple[Any, Any], Dict[str, Dict[str, float]]],
               ) -> None:
    print(f"\n{'='*70}\nwer grid, {split} (mean ± std over seeds)\n{'='*70}")
    header = "alpha\\beta".ljust(10) + "".join(f"{b!s:>16}" for b in betas)
    print(header)
    for a in alphas:
        row = f"{a!s:<10}"
        for b in betas:
            s = summary.get((a, b), {}).get(split)
            row += f"{s['mean']:>9.4f}±{s['std']:<5.4f}" if s else f"{'n/a':>16}"
        print(row)


def print_flat(cell_names: List[str], split: str,
               summary: Dict[Any, Dict[str, Dict[str, float]]],
               cells_by_name: Dict[str, Any]) -> None:
    """Generic single-line-per-cell listing, used whenever --name-regex
    doesn't capture exactly {alpha, beta} (e.g. a 1-parameter sweep like
    peak-preserving's gamma or peak-capping's alpha-as-cap)."""
    print(f"\n{'='*70}\nwer, {split} (mean ± std over seeds)\n{'='*70}")
    for name in cell_names:
        s = summary.get(cells_by_name[name], {}).get(split)
        if s:
            print(f"  {name:<40} {s['mean']:.4f} ± {s['std']:.4f} "
                  f"(n={len(s['values'])})")
        else:
            print(f"  {name:<40} n/a")


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    name_re = re.compile(args.name_regex)

    checkpoints = find_checkpoints(Path(args.checkpoint_top_dir), args.pattern)
    if not checkpoints:
        print(f"No checkpoints matched --pattern {args.pattern!r} under "
              f"{args.checkpoint_top_dir}")
        return
    print(f"Found {len(checkpoints)} checkpoint(s) matching {args.pattern!r}")

    # cell_results[(alpha, beta)][split] = [wer, wer, ...] over seeds
    cell_results: Dict[Tuple[Any, Any], Dict[str, List[float]]] = {}
    ungrouped: List[Dict[str, Any]] = []
    all_raw: Dict[str, List[Dict[str, Any]]] = {}
    n_skipped = 0

    for i, checkpoint_dir in enumerate(checkpoints, 1):
        run_name = checkpoint_dir.parent.name
        log_path = log_dir / f"{run_name}.log"
        tag = f"[{i}/{len(checkpoints)} {run_name}]"

        cached = None if args.force else try_load_cached(log_path)
        if cached is not None:
            n_skipped += 1
            print(f"{tag} SKIP (already completed, reusing {log_path})")
            results = cached
        else:
            results = run_one(args.inference_script, checkpoint_dir, log_path, tag)

        all_raw[run_name] = results

        m = name_re.search(run_name)
        for r in results:
            if m:
                # Generic grouping key: every named regex group except
                # "seed", sorted by group name for a stable, hashable
                # key -- works for the 2-param alpha/beta case (SETS)
                # and any 1-param case (e.g. peak-preserving's gamma,
                # peak-capping's alpha-as-cap) without code changes.
                cell = tuple(sorted(
                    (k, _unfmt(v)) for k, v in m.groupdict().items()
                    if k != "seed"))
                cell_results.setdefault(cell, {}).setdefault(
                    r["test_split"], []).append(r["wer"])
            else:
                ungrouped.append({**r, "run_name": run_name})

        # Write incrementally -- see run_train_grid_seed.py for why.
        summary_path = log_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "raw": all_raw,
                "grid": {
                    "_".join(f"{k}={v}" for k, v in cell): v
                    for cell, v in cell_results.items()
                },
                "ungrouped": ungrouped,
            }, f, indent=2, default=str)

    print(f"\n{n_skipped}/{len(checkpoints)} checkpoints were skipped "
          f"(already-completed logs reused).")

    if cell_results:
        param_names = {k for cell in cell_results for k, _ in cell}
        summary = summarize_grid(cell_results)
        if param_names == {"alpha", "beta"}:
            alphas = sorted(
                {v for c in cell_results for k, v in c if k == "alpha"},
                key=str)
            betas = sorted(
                {v for c in cell_results for k, v in c if k == "beta"},
                key=str)
            grid_summary = {
                (dict(c)["alpha"], dict(c)["beta"]): s
                for c, s in summary.items()
            }
            for split in ("test-clean", "test-other"):
                print_grid(alphas, betas, split, grid_summary)
        else:
            cell_names = {
                "_".join(f"{k}={v}" for k, v in cell): cell
                for cell in cell_results
            }
            for split in ("test-clean", "test-other"):
                print_flat(sorted(cell_names), split, summary, cell_names)
    if ungrouped:
        print(f"\n{len(ungrouped)} result(s) didn't match --name-regex "
              f"(reported individually in summary.json under 'ungrouped').")

    print(f"\nFull results + summary written to {log_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
