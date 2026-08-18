#!/usr/bin/env python3
"""Runs a training shell script (e.g. run_train_fixed.sh / run_train_dynamic.sh)
multiple times with different --seed values, and summarizes the results.

Why this exists: GPU training here is NOT bit-identical even for a fixed
seed (cudnn.benchmark autotuning, non-deterministic CUDA reduction kernels
like scatter_add_, multi-worker DataLoader interleaving), and on top of
that, --seed changes weight init AND the length-bucketing/dynamic-batching
sample order (see wav2vec2_finetuning_sets.py's --seed). So comparing two
batching strategies (or any two configs) from a single run each can't
distinguish a real effect from run-to-run training noise -- you need
several independent repeats per config and to look at mean +/- std, not
a single number.

Deliberately does NOT reimplement the training invocation in Python: the
.sh scripts are still the single source of truth for hyperparameters (only
change those, not this file, to tweak a flag). This script only adds a
thin orchestration layer on top: run each script N times (only the seed
varies -- each .sh accepts it as `bash <script> <seed>`), capture output,
and parse the final metrics HF Trainer prints (which are literal Python
dict reprs, e.g. `{'eval_wer': 0.21, ...}` -- reliably parsed with
`ast.literal_eval`, unlike trying to regex/awk this out of raw log text).

Each script's own N seeds always run SEQUENTIALLY (they'd contend for the
same GPU memory otherwise, defeating the point) -- only DIFFERENT scripts
can run in parallel, one per GPU, via --gpus.

Usage:
    # One script at a time (e.g. two terminals, each with its own
    # `source set_config.sh` / DEVICE_ID -- no --gpus needed):
    python run_multi_seed.py run_train_fixed.sh
    python run_multi_seed.py run_train_dynamic.sh

    # Both at once, one call, each pinned to its own GPU in parallel:
    python run_multi_seed.py run_train_fixed.sh run_train_dynamic.sh \\
        --gpus 0 1

    python run_multi_seed.py run_train_dynamic.sh --seeds 0 1 2 3 4
"""

import argparse
import ast
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scripts", nargs="+",
        help="One or more training shell scripts to run, each accepting "
             "seed as `bash <script> <seed>` (see run_train_fixed.sh / "
             "run_train_dynamic.sh).")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Seeds to run each script with (default: 0 1 2).")
    parser.add_argument(
        "--log-dir", type=str, default="multi_seed_logs",
        help="Directory for per-run raw logs and the combined summary "
             "(default: multi_seed_logs).")
    parser.add_argument(
        "--gpus", type=str, nargs="+", default=None,
        help="One GPU id per --scripts entry (same order), e.g. "
             "`--gpus 0 1` for two scripts. Runs each script's full "
             "--seeds sweep in its own thread with CUDA_VISIBLE_DEVICES "
             "pinned to that GPU, all scripts running IN PARALLEL against "
             "each other (each script's own seeds still run one at a time "
             "-- see module docstring for why). Omit to run scripts one "
             "at a time, sequentially, inheriting whatever "
             "CUDA_VISIBLE_DEVICES is already set in the environment "
             "(e.g. from `source set_config.sh`).")
    args = parser.parse_args()
    if args.gpus is not None and len(args.gpus) != len(args.scripts):
        parser.error(
            f"--gpus has {len(args.gpus)} value(s) but {len(args.scripts)} "
            f"script(s) were given -- need exactly one GPU id per script.")
    return args


def run_one(script: str, seed: int, log_path: Path,
           gpu: Optional[str], tag: str) -> Dict[str, Any]:
    """Runs `bash <script> <seed>`, streaming output live (each line
    prefixed with `tag` so parallel scripts' output stays distinguishable)
    while also capturing it to `log_path` and scanning it for the final HF
    Trainer metrics dicts.

    Args:
        gpu: If given, sets CUDA_VISIBLE_DEVICES=<gpu> for this subprocess
            only (via a copy of the current environment -- doesn't affect
            the orchestrator's own env or other concurrent runs). If None,
            the subprocess inherits the ambient environment unchanged.
        tag: Short label prefixed to every streamed/logged line, e.g.
            "[run_train_fixed.sh seed=0]" -- mainly useful when multiple
            scripts are interleaving output in --gpus mode.

    Returns a dict with at least "returncode" and "wall_time_s", plus
    "eval_metrics" / "train_summary" (the last matching dict literal seen
    in the output, if any -- "last" because Trainer logs an eval dict at
    every --eval_steps, and the last one is the final/best-comparable one;
    "train_summary" only ever gets logged once, at the very end).
    """
    print(f"\n{'='*70}\n>>> {tag}\n{'='*70}")
    env = None
    if gpu is not None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        ["bash", script, str(seed)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    eval_metrics: Optional[Dict[str, Any]] = None
    train_summary: Optional[Dict[str, Any]] = None

    with open(log_path, "w") as log_file:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(f"{tag} {line}")
            log_file.write(line)

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
        "script": script,
        "seed": seed,
        "gpu": gpu,
        "returncode": returncode,
        "wall_time_s": wall_time_s,
        "eval_metrics": eval_metrics,
        "train_summary": train_summary,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Builds mean/std across successful runs for the metrics that matter
    for comparing configs: eval_wer, eval_loss, train_runtime.

    Runs that crashed (non-zero returncode) or never printed a final
    eval_metrics dict are excluded from the statistics but still listed
    per-run, so a crash is visible rather than silently dropped.
    """
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


def print_summary(script: str, results: List[Dict[str, Any]],
                  summary: Dict[str, Any]) -> None:
    print(f"\n{'-'*70}\nSummary for {script} "
          f"({summary['n_ok']}/{summary['n_total']} runs OK)\n{'-'*70}")
    for r in results:
        status = "OK" if r["returncode"] == 0 else f"FAILED(rc={r['returncode']})"
        wer = (r["eval_metrics"] or {}).get("eval_wer")
        wer_str = f"{wer:.4f}" if wer is not None else "n/a"
        print(f"  seed={r['seed']:<3} {status:<14} eval_wer={wer_str}  "
              f"wall_time={r['wall_time_s']:.0f}s")
    for key in ("eval_wer", "eval_loss", "train_runtime"):
        if key in summary:
            s = summary[key]
            print(f"  {key}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"(n={len(s['values'])})")


def run_script_sweep(
    script: str, seeds: List[int], log_dir: Path, gpu: Optional[str],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Runs all `seeds` for one script, sequentially (see module docstring
    for why seeds within a script are never parallelized), and returns
    (script, results, summary).
    """
    script_name = Path(script).stem
    results = []
    for seed in seeds:
        log_path = log_dir / f"{script_name}_seed{seed}.log"
        gpu_tag = f" gpu={gpu}" if gpu is not None else ""
        tag = f"[{script_name} seed={seed}{gpu_tag}]"
        result = run_one(script, seed, log_path, gpu, tag)
        results.append(result)
        if result["returncode"] != 0:
            print(f"!!! {tag} exited with code {result['returncode']} "
                  f"-- see {log_path}")
    summary = summarize(results)
    print_summary(script, results, summary)
    return script, results, summary


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    all_summaries: Dict[str, Any] = {}

    if args.gpus is not None:
        # One script per GPU, all scripts running in parallel against each
        # other (each script's own seeds still sequential within its
        # thread).
        with ThreadPoolExecutor(max_workers=len(args.scripts)) as pool:
            futures = [
                pool.submit(run_script_sweep, script, args.seeds, log_dir, gpu)
                for script, gpu in zip(args.scripts, args.gpus)
            ]
            for future in as_completed(futures):
                script, results, summary = future.result()
                all_results[script] = results
                all_summaries[script] = summary
    else:
        # No GPU pinning requested: run scripts one at a time, in order,
        # inheriting whatever CUDA_VISIBLE_DEVICES is already in the
        # environment.
        for script in args.scripts:
            script, results, summary = run_script_sweep(
                script, args.seeds, log_dir, gpu=None)
            all_results[script] = results
            all_summaries[script] = summary

    summary_path = log_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"results": all_results, "summary": all_summaries}, f, indent=2)
    print(f"\nFull results + summary written to {summary_path}")


if __name__ == "__main__":
    main()
