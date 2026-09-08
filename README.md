# selective_estimated_target_smoothing

Experiment scripts for fine-tuning wav2vec2 with the SHC loss (alpha/beta
smoothing) on LibriSpeech, and evaluating the results. Training code lives
in `cognitive_workflow_kit`; this repo holds the run scripts,
sweep orchestrators, and their logs.

## Setup

Steps 1–5 are one-time. Step 6 is per-shell — repeat it in every new
terminal before running anything below.

### 1. Create and activate the conda environment

```bash
conda create --name py3_12_sets python=3.12
conda activate py3_12_sets
```

3.11 works equally well. Nothing in this repo or in CWK constrains the
version (CWK's `pyproject.toml` only asks for `>=3.7`); 3.12 is chosen to
match the system Python. If `conda` isn't on your PATH, install Miniconda
first — or use `python3 -m venv` instead, since nothing here depends on
conda specifically.

### 2. Install PyTorch and torchaudio

Check your CUDA version with `nvidia-smi`, then pick the matching command
from https://pytorch.org/get-started/locally/. The RTX 5090s on this machine
are Blackwell (sm_120) and require a **CUDA 12.8+** build — an older wheel
installs cleanly and then fails at runtime with "no kernel image is
available for execution on the device":

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install torch and torchaudio together so their versions stay matched.

### 3. Install the remaining dependencies

```bash
pip install numpy                     # imported directly by both entry points
pip install transformers accelerate   # model, processor, Trainer
pip install evaluate jiwer            # evaluate.load("wer") needs jiwer
pip install webdataset                # sharded .tar training data
pip install soundfile                 # FLAC decoding
pip install sentencepiece             # unigram vocab
pip install flashlight-text           # torchaudio ctc_decoder backend
```

That is every third-party import in this repo and in the `common` helpers it
uses, plus three that are never imported by name: `accelerate` (required by
HF `Trainer`), `jiwer` (the backend `evaluate.load("wer")` loads), and
`flashlight-text`. `numpy` also arrives transitively with `transformers`,
but it is listed because the scripts import it directly.

`flashlight-text` is only needed by `wav2vec2_inference.py`'s beam search —
`torchaudio.models.decoder.ctc_decoder` is a thin wrapper over it and raises
without it. KenLM is *not* needed: the decoder is constructed with
`lm=None` and `lexicon=None`.

### 4. Install the CWK package

The two entry-point scripts import from the sibling `cognitive_workflow_kit`
repo in two different ways, and each needs its own step:

```bash
cd <your cognitive_workflow_kit checkout>   # this path becomes CWK_HOME in step 5
pip install -e .
cd -
```

That covers `from cwk.loss.pytorch import shc_loss` — the SHC loss itself.
The other import, `from common import sample_util`, lives in that repo's
`scripts/` directory, which its `pyproject.toml` deliberately excludes from
the package (`include = ["cwk*"]`). It resolves via `PYTHONPATH` in step 6
instead, which is why sourcing `set_config.sh` isn't optional.

### 5. Point the repo at this machine's paths

**This is the only file you edit per machine.** No absolute path is baked
into any tracked script, so a fresh `git clone` elsewhere needs this step
and nothing else:

```bash
cp config.local.sh.example config.local.sh
# then edit config.local.sh
```

It defines four values:

| Variable | What it is |
|---|---|
| `CWK_HOME` | The `cognitive_workflow_kit` checkout from step 4 — supplies `cwk`, `common`, and the SPM vocabularies under its `resources/spm/` |
| `DB_TOP_DIR` | Dataset root, holding `libri_light_finetuning/webdataset/{1h,10h}` and `librispeech/webdataset/{train-clean-100,test-clean,test-other,...}`, each a directory of `shard-*.tar` files |
| `CHECKPOINT_TOP_DIR` | Where training writes checkpoints. Prefer a local disk over NFS — these are written often enough that network latency shows up in step time |
| `PYTHON_BIN` | `bin/` of the conda env from step 1, for `queue_*.sh` — a `nohup`-style launch doesn't inherit an activated env. Leave empty to use whatever `python` is on PATH |

`config.local.sh` is gitignored, so each machine keeps its own and nothing
here conflicts across clones. `set_config.sh` sources it, and the Python
scripts read the same values through `repo_config.py`, which is what makes
the `--db_top_dir` / `--resource_top_dir` / `--checkpoint_top_dir` defaults
correct without anyone passing them.

### 6. Per-shell environment

```bash
source set_config.sh
```

Loads `config.local.sh`, then sets `PYTHONPATH` (via the CWK repo's
`setup_path.sh`), `CUDA_VISIBLE_DEVICES`, and a few NCCL/allocator env vars.
It stops with an explicit message if `config.local.sh` is missing, so a
fresh clone tells you what to do rather than failing later on a wrong path.
To pick a specific GPU, set `DEVICE_ID` first:

```bash
export DEVICE_ID=1
source set_config.sh
```

### 7. Verify the install

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from cwk.loss.pytorch import shc_loss, shc_loss_util; print('cwk ok')"
python -c "from common import sample_util; print('common ok')"
python -c "from torchaudio.models import decoder; decoder.ctc_decoder; print('decoder ok')"
python -c "import repo_config, os; [print(('ok  ' if os.path.isdir(p) else 'MISSING '), p) for p in (repo_config.DB_TOP_DIR, repo_config.RESOURCE_TOP_DIR, repo_config.CHECKPOINT_TOP_DIR)]"
```

If the third fails, `set_config.sh` wasn't sourced in this shell. If the
fourth fails, `flashlight-text` is missing — training still works, only
beam-search inference is affected. The last one is the check that the paths
in `config.local.sh` actually exist on this machine.

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

### The 1hr baseline, written out in full

`run_train_dynamic.sh` is the 1hr baseline. Spelled out flag by flag — copy
this as the starting point for a new config:

```bash
python wav2vec2_finetuning_sets.py \
    --alpha=0.0 \
    --beta=0.0 \
    --vocab_size 32 \
    --finetune_profile=libri_light_1hr \
    --dynamic_batching \
    --max_batch_audio_len 6400000 \
    --max_sample_audio_len 480000 \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers \
    --seed 0
```

| Flag | Meaning |
|---|---|
| `--alpha=0.0 --beta=0.0` | Smoothing off — this is what makes it the *baseline*. `alpha > 0` is the switch (`smoothing_enabled` in `shc_loss.py`), so at `alpha=0` the loss is plain CTC and `beta`, which only chooses the mixing distribution, has no effect |
| `--vocab_size 32` | Use the SentencePiece unigram-32 vocab (`librispeech_unigram_32.model` under `--resource_top_dir`). Omit for wav2vec2's own character tokenizer |
| `--finetune_profile=libri_light_1hr` | The 1h Libri-Light set, plus the schedule sized for it: warmup 1000, **max_steps 2000**, eval every 500 |
| `--dynamic_batching` | Fill each batch to an audio-length budget instead of a fixed example count — fewer padding-wasted frames when utterance lengths vary |
| `--max_batch_audio_len 6400000` | That budget, in waveform samples: 6.4M ÷ 16 kHz ≈ **400 s of audio per batch** |
| `--max_sample_audio_len 480000` | Drop any utterance longer than **30 s** (480000 ÷ 16 kHz) — a safety net against mis-segmented outliers. Omit to disable |
| `--dataloader_num_workers 4` | Decode audio on 4 CPU workers in parallel with GPU compute |
| `--dataloader_persistent_workers` | Keep those workers alive between epochs instead of respawning them |
| `--seed 0` | Weight init + shuffle order. Also tagged into the run name, so seeds never overwrite each other |

The run above lands in
`$CHECKPOINT_TOP_DIR/libri_light_1hr_shc_2000steps_alpha_0p0_beta_0p0_unigram_32_dynbatch6400000_seed0`
and takes about an hour (measured: 58–60 min across the 15 runs in
`grid_logs_1hr`, on the previous 4090-class machine).

One caveat when copying this: `--smoothing_space=class` is a no-op at
`alpha=0` — every space runs the same ops — but it still appends
`_classspace` to the run name, so a baseline carrying it will not group with
the other baselines under `run_inference_sweep.py --pattern`. Leave it off
unless `alpha > 0`.

Checkpoints are written under `--checkpoint_top_dir`
(`$CHECKPOINT_TOP_DIR` by default), one subdirectory per run,
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
`--checkpoint-top-dir` (default `$CHECKPOINT_TOP_DIR`); each
match's highest-numbered `checkpoint-N` is evaluated. If the directory
name contains `alpha_..._beta_..._seed...` (as the training scripts name
them), results are grouped into an alpha × beta grid — printed separately
for test-clean and test-other. Also resumable/skips completed checkpoints,
same as `run_train_grid_seed.py`.

## Files

| File | Purpose |
|---|---|
| `config.local.sh.example` | Template for the per-machine paths; copy to `config.local.sh` (gitignored) and edit |
| `set_config.sh` | Per-shell env setup — sources `config.local.sh`, sets PYTHONPATH and GPU selection |
| `repo_config.py` | Python-side view of those paths; every script's directory defaults come from here |
| `wav2vec2_finetuning_sets.py` | Training entry point |
| `wav2vec2_inference.py` | Evaluation entry point (WER via pipeline or beam search) |
| `run_train_fixed.sh`, `run_train_dynamic.sh` | Single training runs |
| `run_train_dynamic_grid.sh` | One grid cell (`<alpha> <beta> <seed> [profile]`) |
| `run_train_fixed_*hr_alpha_*_beta_*.sh` | One-off named configs |
| `run_multi_seed.py` | Multi-seed comparison of a few named configs |
| `run_train_grid_seed.py` | Multi-seed alpha × beta sweep |
| `run_inference.sh` | Final WER for one checkpoint (both test splits) |
| `run_inference_sweep.py` | Final WER for every checkpoint from a sweep |
| `analysis/grid_table.py` | Alpha × beta results table from a grid_logs_* dir, in-progress seeds included |
