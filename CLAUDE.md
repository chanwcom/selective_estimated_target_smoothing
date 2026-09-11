# Working notes for this repo

## READ FIRST: every result produced before 2026-09-11 07:11 KST is contaminated

Two bugs in the CTC target construction were found and fixed on 2026-09-11.
Both sat upstream of every smoothing method, so **plain CTC, textbook label
smoothing, C-SETS, L-SETS, peak-capping, peak-preserving, entropy-matched and
AS were all affected.** Any number produced before the fix is not comparable
with any number produced after it.

### Bug 1 -- padding dominated the alignment-posterior normalization

`calculate_alpha_beta` floors padded label positions at the *finite* constant
`LOG_0 = -706.89` (not `-inf`), so they enter
`log_gamma = log_alpha + log_beta` at `2 * LOG_0 = -1413.79`. Normalizing with
`logsumexp` over the label axis was only safe while every real alignment
scored above that floor. It does not early in training: a long utterance sums
`T` per-frame log-probabilities, so at roughly `T >= 800` with a not-yet-
confident head the best real path drops below the floor. Padding then wins the
logsumexp, every valid position normalizes to about -140, `exp()` flushes it
to zero, and -- padded labels being `clamp(min=0)` = blank -- the utterance's
target becomes **"blank with probability 1 at every frame"**.

At `--max_sample_audio_len 480000` (30 s) `T` reaches 1500, so this fired on
long utterances for roughly the first 150-200 steps of every run.

Fixed in `ShcLoss.forward` by masking padded label positions to `-inf` before
the logsumexp (and normalizing in float32). The fix is bit-identical wherever
the failure did not occur.

Note the masking cannot be pushed into `calculate_alpha_beta` as a blanket
`-inf`: padded *time* frames have every label position masked, so an all-`-inf`
row would make `logsumexp` return `-inf` and the normalization produce `NaN`.
That variant was tried and breaks 14 tests. `LOG_0` being finite is deliberate.

### Bug 2 -- label-space smoothing spread its uniform over padding

In `smoothing_space="label"`, SETS's uniform component used the batch's padded
label width, so `alpha * (L - S) / L` of the smoothing mass landed on positions
that do not exist and scattered onto blank. A sample's target therefore
depended on how long its batch-mates happened to be (measured: same utterance
drifting by ~4e-2 in target probability). Fixed by passing `axis_lens` so the
uniform is confined to each sample's own label length. Only bites at `beta < 1`;
class space was never affected (the class axis has no padding).

### Where the contaminated artifacts live

| What | Where |
|---|---|
| Pre-fix checkpoints (31 runs, 33 GB) | `~/models_PREBUGFIX_before_20260911_0711/` |
| Pre-fix AS sweep logs | `prebugfix_logs/` in this repo |
| Pre-fix logs still in place | `grid_logs_1hr*`, `grid_logs_10hr*` **except** the dirs listed as post-fix below |

Everything under `~/models/` is post-fix. Anything in
`~/models_PREBUGFIX_before_20260911_0711/` is not.

**Do not merge pre- and post-fix numbers into one table.** If you need a
baseline to compare against, re-run it rather than reusing an old log.

### Post-fix results so far (1hr, seed 0 only unless noted)

- `grid_logs_1hr_as_a0p05` / `a0p1` / `a0p15` / `a0p2` / `a0p25` / `a0p3` /
  `a0p35` / `a0p4` -- AS (`--alpha_mode active_support`)
- `grid_logs_1hr_ls_a0p1_postfix` -- textbook LS alpha=0.1, queued

### Regression tests

`cwk/loss/pytorch/shc_loss_bug_test.py` covers both bugs
(`test_long_utterance_target_is_not_all_blank`,
`test_label_space_target_is_invariant_to_batch_mate_length`). It uses bare
pytest-style functions, and pytest is not installed in `py3_12_sets` -- run it
as `python shc_loss_bug_test.py`. The rest of the suite is `unittest`:
`python -m unittest shc_loss_test` etc.

## Environment

`source set_config.sh` before anything (it needs the gitignored
`config.local.sh`). Machine-specific paths: CWK checkout and dataset root live
on the NAS and are shared; `~/models`, `~/miniconda3` are per-machine.

`set_config.sh` reaches CWK's `setup_path.sh`, which expands `$PYTHONPATH`
unguarded -- that aborts under `set -u` in a detached (nohup) shell. Queue
scripts export `PYTHONPATH="${PYTHONPATH:-}"` first; copy that if you write a
new one.

## Shared working tree

This repo sits on a NAS mount and is worked on from more than one machine, as
a single clone rather than one clone per machine. Two sessions editing the
same file will clobber each other, and `.git` is shared. Coordinate before
editing source; prefer per-machine `--log-dir` values (log filenames are
deterministic, and `run_train_grid_seed.py` read-modify-writes `summary.json`).
