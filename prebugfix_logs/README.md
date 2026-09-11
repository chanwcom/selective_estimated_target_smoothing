# Pre-bugfix sweep logs -- quarantined 2026-09-11

These AS sweep logs were produced before the two CTC target-construction bugs
were fixed (2026-09-11 07:11 KST). They are kept only as evidence of what the
bugs did -- do not merge these numbers with post-fix results.

They live here rather than in their original grid_logs_* directories because
run_train_grid_seed.py reuses any log that already contains both an eval_wer
and a train_runtime, so leaving them in place would make a re-run silently
skip the cell and hand back the contaminated result.

Matching checkpoints: ~/models_PREBUGFIX_before_20260911_0711/
Full write-up: CLAUDE.md in the repo root.
