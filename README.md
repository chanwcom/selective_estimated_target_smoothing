# selective_estimated_target_smoothing

Experiment scripts for fine-tuning wav2vec2 with the SHC loss (alpha/beta
smoothing) on LibriSpeech, and evaluating the results. Training code lives
in `cognitive_workflow_kit_emnlp_2026`; this repo holds the run scripts,
sweep orchestrators, and their logs.

## Setup

```bash
source set_config.sh
```

Sets `PYTHONPATH` (via the CWK repo's `setup_path.sh`), `CUDA_VISIBLE_DEVICES`,
and a few NCCL/allocator env vars. To pick a specific GPU, set `DEVICE_ID`
first:

```bash
export DEVICE_ID=1
source set_config.sh
```

Long jobs (anything below can run for hours to days) should be started
inside `tmux` so they survive a dropped SSH connection:

```bash
tmux new -s <name>
# ... run your command ...
# Ctrl+b, d      detach (job keeps running)
# tmux attach -t <name>   reattach later
```

## 1. Single training run

```bash
bash run_train_fixed.sh [seed]      # fixed batch_size=24, bucketing off, 25s length cap
bash run_train_dynamic.sh [seed]    # dynamic (length-budget) batching
```

Both wrap `wav2vec2_finetuning_sets.py`. `seed` is an optional positional
arg (default 42) that also feeds the auto-generated `--run_name`, so
different seeds never overwrite each other's checkpoints.

Other one-off configs (e.g. `run_train_fixed_10hr_alpha_0p02_beta_1p0.sh`)
follow the same pattern — same structure as `run_train_fixed.sh`, just a
different `--finetune_profile` and/or `--alpha`/`--beta`. Copy one of these
to make a new one-off config.

**Key flags** (see each script or `python wav2vec2_finetuning_sets.py --help`
for the full list):

| Flag | Meaning |
|---|---|
| `--alpha`, `--beta` | SHC smoothing coefficients |
| `--finetune_profile` | Fine-tuning set: `libri_light_1hr`, `libri_light_10hr`, `libri_speech_clean_100hr`, `libri_speech_full_960hr` |
| `--dynamic_batching` + `--max_batch_audio_len` | Length-budget batching instead of fixed batch size |
| `--max_sample_audio_len` | Drop any utterance longer than this many samples (safety cap) |
| `--seed` | Weight init + data shuffle order (also tags `--run_name`) |
| `--dataloader_num_workers` | Overlaps CPU audio decode with GPU compute |

Checkpoints are written under `--checkpoint_top_dir`
(`/mnt/data/home/chanwcom/models` by default), one subdirectory per run,
named automatically from the flags above (profile, alpha, beta, vocab
size, batching mode, seed) — different configs never collide.

## 2. Comparing a couple of named configs across seeds

```bash
python run_multi_seed.py run_train_fixed.sh run_train_dynamic.sh
python run_multi_seed.py run_train_dynamic.sh --seeds 0 1 2 3 4
python run_multi_seed.py run_train_fixed.sh run_train_dynamic.sh --gpus 0 1   # run in parallel, one GPU each
```

Runs each `.sh` script once per seed (default seeds `0 1 2`), streams
output live, and reports per-seed results plus mean ± std for `eval_wer`,
`eval_loss`, `train_runtime`. Logs and a `summary.json` go to
`--log-dir` (default `multi_seed_logs`).

Why seeds, not a single run: GPU training here isn't bit-identical even
with a fixed seed (cudnn autotuning, non-deterministic CUDA reduction
kernels), and `--seed` also changes weight init and batch order — so one
run per config can't tell a real effect from run-to-run noise.

## 3. Sweeping alpha × beta

```bash
python run_train_grid_seed.py run_train_dynamic_grid.sh \
    --alphas 0.01 0.02 0.03 0.04 \
    --betas 0.0 0.25 0.5 0.75 1.0 \
    --seeds 0 1 2 \
    --profile libri_light_1hr \
    --log-dir grid_logs_1hr
```

Runs every (alpha, beta, seed) combination, printing a grid of mean ± std
`eval_wer` per cell once done. `--profile` switches the fine-tuning set —
use a different `--log-dir` per profile to keep results separate:

```bash
python run_train_grid_seed.py run_train_dynamic_grid.sh \
    --alphas ... --betas ... \
    --profile libri_light_10hr --log-dir grid_logs_10hr
```

**This is a multi-hour-to-multi-day job.** Cells with an already-complete
log are skipped automatically on rerun (pass `--force` to redo everything)
— if it gets interrupted, just run the same command again to resume.

To split a sweep across two GPUs by hand, run two instances with disjoint
`--alphas`/`--betas` in separate terminals (each with its own
`DEVICE_ID`/`source set_config.sh`) pointed at the same `--log-dir`.

## 4. Final evaluation (WER via beam search)

The numbers `run_train_grid_seed.py` reports during training are a fast
proxy only: greedy-decoded, test-clean only. The real final number comes
from beam search over **both** test-clean and test-other:

```bash
bash run_inference.sh <checkpoint_dir>
```

Runs `wav2vec2_inference.py` with beam search (`beam_size=20`) on
test-clean and test-other. Beam size 20 is a deliberate speed/quality
tradeoff (CTC beam search without an LM has fast-diminishing returns past
~20) — pass a larger `--beam_size` directly to `wav2vec2_inference.py` if
you want to check whether it changes the ranking between configs.

To evaluate every checkpoint from a sweep at once:

```bash
python run_inference_sweep.py \
    --pattern "libri_light_1hr_shc_2500steps_alpha_*_beta_*_unigram_32_dynbatch6400000_seed*" \
    --log-dir inference_logs_1hr
```

`--pattern` is matched against run directory names under
`--checkpoint-top-dir` (default `/mnt/data/home/chanwcom/models`); each
match's highest-numbered `checkpoint-N` is evaluated. If the directory
name contains `alpha_..._beta_..._seed...` (as the training scripts name
them), results are grouped into an alpha × beta grid — printed separately
for test-clean and test-other. Also resumable/skips completed checkpoints,
same as `run_train_grid_seed.py`.

## Files

| File | Purpose |
|---|---|
| `set_config.sh` | Env setup (PYTHONPATH, GPU selection) |
| `wav2vec2_finetuning_sets.py` | Training entry point |
| `wav2vec2_inference.py` | Evaluation entry point (WER via pipeline or beam search) |
| `run_train_fixed.sh`, `run_train_dynamic.sh` | Single training runs |
| `run_train_dynamic_grid.sh` | One grid cell (`<alpha> <beta> <seed> [profile]`) |
| `run_train_fixed_*hr_alpha_*_beta_*.sh` | One-off named configs |
| `run_multi_seed.py` | Multi-seed comparison of a few named configs |
| `run_train_grid_seed.py` | Multi-seed alpha × beta sweep |
| `run_inference.sh` | Final WER for one checkpoint (both test splits) |
| `run_inference_sweep.py` | Final WER for every checkpoint from a sweep |
