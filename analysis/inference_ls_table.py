#!/usr/bin/env python3
"""Summarizes the classical-LS inference sweep next to the training-time eval.

Three numbers per (profile, alpha) cell, all on the same checkpoints:

  train   -- the LAST eval_wer in the training log, not the minimum. The
             minimum over eight eval points is a best-of-8 statistic and
             is biased low by roughly one SE; the final value is what the
             model actually ends at, which is what the inference runs
             below decode from. Greedy, test-clean only.
  greedy  -- run_inference_ls_sweep.sh with --decoder=pipeline.
  beam40  -- same, --decoder=beam_search --beam_size 40 (log_add=True).

`train` and `greedy` decode the same checkpoint the same way, so a gap
between them is the evaluation setup (dataset iteration, reference
normalization), not the decoder. `beam40` minus `greedy` is the decoder
alone.

Usage:
    python analysis/inference_ls_table.py
    python analysis/inference_ls_table.py --log-dir inference_logs_ls
"""

import argparse
import ast
import collections
import glob
import math
import os
import re
import statistics as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_LOG_DIRS = {
    "libri_light_1hr": os.path.join(REPO, "grid_logs_1hr_ls_postfix_4090"),
    "libri_light_10hr": os.path.join(REPO, "grid_logs_10hr_ls_postfix_4090"),
}


def parse_train_logs():
    """Final (not best) eval_wer per (profile, alpha, seed)."""
    out = {}
    for profile, d in TRAIN_LOG_DIRS.items():
        for path in glob.glob(os.path.join(d, "libri_light_*alpha*.log")):
            text = open(path).read()
            if "'train_runtime'" not in text:
                continue
            wers = re.findall(r"'eval_wer': ([0-9.]+)", text)
            if not wers:
                continue
            base = os.path.basename(path)
            alpha = float(re.search(r"alpha(0p\d+)", base).group(1).replace("p", "."))
            seed = int(re.search(r"seed(\d)", base).group(1))
            out[(profile, alpha, seed)] = float(wers[-1])
    return out


def parse_inference_logs(log_dir):
    """WER per (profile, alpha, seed, decoder, split)."""
    out = {}
    for path in glob.glob(os.path.join(log_dir, "libri_light_*.log")):
        text = open(path).read()
        match = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{") and "'wer':" in line:
                match = line
        if match is None:
            continue
        try:
            result = ast.literal_eval(match)
        except (ValueError, SyntaxError):
            continue
        base = os.path.basename(path)
        m = re.match(
            r"(libri_light_\d+hr)_alpha([0-9p]+)_seed(\d)_(beam_search|pipeline)_(test-\w+)\.log",
            base)
        if not m:
            continue
        profile, alpha, seed, decoder, split = m.groups()
        key = (profile, float(alpha.replace("p", ".")), int(seed), decoder, split)
        out[key] = result["wer"]
    return out


def pooled_sd(cells):
    """Pooled within-cell sd over {key: [values]}, and its dof."""
    ss, dof = 0.0, 0
    for values in cells.values():
        if len(values) > 1:
            mean = st.mean(values)
            ss += sum((v - mean) ** 2 for v in values)
            dof += len(values) - 1
    return (math.sqrt(ss / dof), dof) if dof else (float("nan"), 0)


def t_two_sided_p(t, dof):
    """Two-sided p for Student's t, via the regularized incomplete beta."""
    t, x, a, b = abs(t), dof / (dof + t * t), dof / 2.0, 0.5

    def betacf(a, b, x):
        c, d = 1.0, 1 - (a + b) * x / (a + 1)
        d = 1 / d if abs(d) > 1e-300 else 1e300
        h = d
        for m in range(1, 400):
            m2 = 2 * m
            num = m * (b - m) * x / ((a - 1 + m2) * (a + m2))
            d, c = 1 / (1 + num * d), 1 + num / c
            h *= d * c
            num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2))
            d, c = 1 / (1 + num * d), 1 + num / c
            delta = d * c
            h *= delta
            if abs(delta - 1) < 3e-16:
                break
        return h

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    if x < (a + 1) / (a + b + 2):
        return math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) * betacf(a, b, x) / a
    return 1 - math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) * betacf(b, a, 1 - x) / b


def fmt(value, width=8):
    return f"{value:>{width}.4f}" if value is not None else "--".rjust(width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=os.path.join(REPO, "inference_logs_ls"))
    args = parser.parse_args()

    train = parse_train_logs()
    infer = parse_inference_logs(args.log_dir)
    print(f"학습 로그 {len(train)}런, 추론 결과 {len(infer)}/240\n")

    COLUMNS = [
        ("train", "train", None, None),
        ("greedy", "infer", "pipeline", "test-clean"),
        ("beam40", "infer", "beam_search", "test-clean"),
        ("greedy", "infer", "pipeline", "test-other"),
        ("beam40", "infer", "beam_search", "test-other"),
    ]

    for profile, title in (("libri_light_1hr", "1hr"), ("libri_light_10hr", "10hr")):
        alphas = sorted({a for (p, a, _) in train if p == profile})
        if not alphas:
            continue
        print("=" * 96)
        print(f"=== 전통 LS {title} ===\n")
        print(f"{'':>6} {'train':>18} {'test-clean':>19} {'test-other':>19}")
        print(f"{'alpha':>6} {'(greedy,clean)':>18} {'greedy':>9}{'beam40':>10} "
              f"{'greedy':>9}{'beam40':>10}   {'n':>3}")
        print("-" * 76)

        cells = collections.defaultdict(dict)
        for alpha in alphas:
            row, counts = [], []
            for _, kind, decoder, split in COLUMNS:
                if kind == "train":
                    values = [train[(profile, alpha, s)] for s in range(5)
                              if (profile, alpha, s) in train]
                else:
                    values = [infer[(profile, alpha, s, decoder, split)] for s in range(5)
                              if (profile, alpha, s, decoder, split) in infer]
                row.append(st.mean(values) if values else None)
                counts.append(len(values))
                if values:
                    cells[(decoder, split)][alpha] = values
            print(f"{alpha:>6.2f} {fmt(row[0], 18)} {fmt(row[1], 9)}{fmt(row[2], 10)} "
                  f"{fmt(row[3], 9)}{fmt(row[4], 10)}   {min(counts):>3}")

        # beam vs greedy, paired per run -- the decoder effect alone.
        print()
        for split in ("test-clean", "test-other"):
            pairs = [(infer[(profile, a, s, "pipeline", split)],
                      infer[(profile, a, s, "beam_search", split)])
                     for a in alphas for s in range(5)
                     if (profile, a, s, "pipeline", split) in infer
                     and (profile, a, s, "beam_search", split) in infer]
            if len(pairs) < 2:
                continue
            deltas = [b - g for g, b in pairs]
            mean, sd = st.mean(deltas), st.stdev(deltas)
            t = mean / (sd / math.sqrt(len(deltas)))
            print(f"  beam40 - greedy [{split}] n={len(deltas):>2}: {mean:+.4f} "
                  f"({mean / st.mean(g for g, _ in pairs):+.1%}), sd {sd:.4f}, "
                  f"대응 t={t:.2f}, p={t_two_sided_p(t, len(deltas) - 1):.4f}, "
                  f"개선된 런 {sum(1 for d in deltas if d < 0)}/{len(deltas)}")

        # Per-alpha significance against the alpha=0 baseline, per column.
        print()
        for _, kind, decoder, split in COLUMNS:
            key = (decoder, split)
            by_alpha = cells.get(key)
            if not by_alpha or 0.0 not in by_alpha:
                continue
            sd, dof = pooled_sd(by_alpha)
            base = by_alpha[0.0]
            best = min((a for a in by_alpha if a > 0), key=lambda a: st.mean(by_alpha[a]))
            values = by_alpha[best]
            se = sd * math.sqrt(1 / len(values) + 1 / len(base))
            t = (st.mean(base) - st.mean(values)) / se
            label = ("train (greedy, test-clean)" if kind == "train"
                     else f"{'beam40' if decoder == 'beam_search' else 'greedy':<6} {split}")
            print(f"  {label:<28} baseline {st.mean(base):.4f} | 최고 a={best:.2f} "
                  f"{st.mean(values):.4f} ({(st.mean(base) - st.mean(values)) / st.mean(base):+.1%}) "
                  f"| pooled sd {sd:.4f} | t={t:.2f} p={t_two_sided_p(t, dof):.4f}")
        print()


if __name__ == "__main__":
    main()
