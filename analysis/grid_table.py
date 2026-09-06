#!/usr/bin/env python3
"""Prints an alpha x beta results table from a grid_logs_* directory.

Complements run_inference_sweep.py: that script computes final WER per
checkpoint, this one aggregates the eval_wer already written into the
training logs by wav2vec2_finetuning_sets.py / run_train_grid_seed.py,
across all three seeds. A cell with all three seeds finished shows the
mean as "X.XXXX(n=3)"; a cell still missing seeds shows the mean of
whatever finished plus which seed is in progress and how far along it is,
e.g. "X.XXXX(n=2,s2 43%)", instead of leaving it blank.

Usage:
    python3 analysis/grid_table.py <log_dir> <profile> <max_steps> \\
        <alphas_csv> <betas_csv>

Example:
    python3 analysis/grid_table.py grid_logs_1hr_class_space \\
        libri_light_1hr 2000 0.01,0.02,0.03 0.25,0.5,0.75,1.0
"""
import re, sys, glob, os

def cell_status(log_dir, alpha, beta, profile, max_steps):
    def fmt(x):
        return str(x).replace(".", "p")
    vals = []
    prog = None
    for seed in (0, 1, 2):
        path = f"{log_dir}/{profile}_alpha{fmt(alpha)}_beta{fmt(beta)}_seed{seed}.log"
        if not os.path.exists(path):
            continue
        text = open(path).read()
        m = re.findall(r"'eval_wer': ([0-9.]+)", text)
        steps = re.findall(rf"(\d+)/{max_steps}", text)
        finished = steps and int(steps[-1]) == max_steps
        if finished and m:
            vals.append(float(m[-1]))
        elif steps:
            prog = (seed, int(steps[-1]), max_steps)
    return vals, prog

def build_table(log_dir, profile, max_steps, alphas, betas):
    print(f"{'a\\\\b':>6s}  " + "  ".join(f"{b:>16.2f}" for b in betas))
    for a in alphas:
        row = []
        for b in betas:
            vals, prog = cell_status(log_dir, a, b, profile, max_steps)
            if len(vals) == 3:
                m = sum(vals)/3
                cell = f"{m:.4f}(n=3)"
            elif vals:
                m = sum(vals)/len(vals)
                tag = f",s{prog[0]} {100*prog[1]//prog[2]}%" if prog else ""
                cell = f"{m:.4f}(n={len(vals)}{tag})"
            elif prog:
                cell = f"(s{prog[0]} {100*prog[1]//prog[2]}%)"
            else:
                cell = "--"
            row.append(cell)
        print(f"{a:>6.2f}  " + "  ".join(f"{c:>16s}" for c in row))

if __name__ == "__main__":
    profile_dir, profile, max_steps, alphas_s, betas_s = sys.argv[1:6]
    alphas = [float(x) for x in alphas_s.split(",")]
    betas = [float(x) for x in betas_s.split(",")]
    build_table(profile_dir, profile, int(max_steps), alphas, betas)
