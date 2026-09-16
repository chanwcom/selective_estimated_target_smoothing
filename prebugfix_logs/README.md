# Pre-bugfix logs -- quarantined 2026-09-11

Everything below this directory was produced before the two CTC
target-construction bugs were fixed (2026-09-11, commit `de53098` in the CWK
repo). They are kept only as evidence of what the bugs did -- **do not merge
these numbers with post-fix results.**

They live here rather than in their original `grid_logs_*` directories because
`run_train_grid_seed.py` reuses any log that already contains both an
`eval_wer` and a `train_runtime`, so leaving them in place would make a re-run
silently skip the cell and hand back the contaminated result.

Matching checkpoints: `~/models_PREBUGFIX_before_20260911_0711/`
Full write-up: `CLAUDE.md` in the repo root.

## Contents

Original directory names are preserved. 812 run logs across 14 sweeps:

| Sweep | Logs | Method |
|---|---|---|
| `grid_logs_1hr` | 78 | L-SETS, label space |
| `grid_logs_1hr_class_space` | 129 | C-SETS |
| `grid_logs_1hr_hybrid` | 183 | L-SETS hybrid |
| `grid_logs_1hr_uniform_ls` | 62 | textbook LS (beta=0) |
| `grid_logs_1hr_entropy_matched` | 11 | entropy matching |
| `grid_logs_1hr_peak_capping` | 15 | PC-SETS |
| `grid_logs_1hr_peak_preserving` | 24 | PP-SETS |
| `grid_logs_1hr_sh` | 4 | SH |
| `grid_logs_10hr` | 64 | L-SETS, label space |
| `grid_logs_10hr_class_space` | 82 | C-SETS |
| `grid_logs_10hr_hybrid` | 64 | L-SETS hybrid |
| `grid_logs_10hr_uniform_ls` | 84 | textbook LS (beta=0) |
| `grid_logs_10hr_entropy_matched` | 1 | entropy matching |
| `grid_logs_10hr_peak_capping` | 11 | PC-SETS |

A handful of logs in `grid_logs_10hr_class_space` and
`grid_logs_10hr_uniform_ls` are truncated mid-run: those cells were still
training when the queues were killed to apply the fix.

`analysis/grid_table.py` still reads these directories if pointed at them
explicitly, which is the intended way to look at them -- for the record of
what the contaminated numbers were, never as a baseline.

## What the contaminated numbers were (1hr, 2000 steps, 3 seeds)

Kept here because the post-fix pilot is only interpretable against them.

| Method | Best WER | alpha | beta |
|---|---|---|---|
| C-SETS | 0.1939 (sd 0.0022) | 0.06 | 0.50 |
| L-SETS hybrid | 0.1976 (sd 0.0031) | 0.06 | 0.25 |
| textbook LS | 0.1980 (sd 0.0009) | 0.11 | -- |
| baseline | 0.2261 (sd 0.0133) | 0 | -- |

10hr best was 0.0984 (L-SETS hybrid, alpha=0.03, beta=0.50); 10hr baseline
0.1030.
