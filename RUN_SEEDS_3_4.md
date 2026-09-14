# Running seeds 3 and 4 of the 10000-step 100hr sweep

`u24` is running seeds **0, 1, 2**. This describes seeds **3 and 4** on
another machine. The two halves share a NAS clone of this repo, so the code
is already whatever `git log` says here — no pull needed if `$CWK_HOME` and
this checkout are the NAS paths. Checkpoints and logs land in per-machine
directories, so nothing collides.

## The grid

| | |
|---|---|
| profile | `libri_speech_clean_100hr` — **10000 steps**, warmup 1000, eval every 500 |
| FAS | `--beta 0.0`, `--fas_eps 1e-10` |
| textbook LS | `--beta 1.0` (bit-identical to uniform LS; see `apply_floored_active_support_smoothing`) |
| alpha | 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 |
| seeds here | **3, 4** |

6 alphas x 2 methods x 2 seeds = **24 cells**, ~5.4 h each.

## Before launching

```bash
cd <this repo>
git pull                      # only if this checkout is NOT the NAS one
grep -A6 'libri_speech_clean_100hr' wav2vec2_finetuning_sets.py | grep max_steps
#   -> max_steps=10000
grep 'CHECKPOINT_TOP_DIR' config.local.sh
#   -> must be this machine's own path; /mnt/.../models/$(hostname) keeps
#      it on the NAS without colliding with u24
```

`config.local.sh` is gitignored, so each machine sets its own.

## Launch

Two GPUs, two runs per GPU. Split the alphas 3/3 so the lanes finish
together.

**Every path below is suffixed with `$(hostname)` on purpose.** This repo is
a single clone on the NAS and both machines run out of it, so a bare
`grid_logs_10k_fas_A` would be the *same directory* on both --
`run_train_grid_seed.py` read-modify-writes `summary.json` in it and the two
machines would clobber each other's. The chain stdout files collide the same
way.

```bash
cd <this repo>
H=$(hostname)
Q(){ echo "GPU=$1 ALPHAS=\"$2\" BETAS=\"$3\" SEEDS=\"$4\" FAS_EPS=1e-10 \
PROFILE=libri_speech_clean_100hr LOG_DIR=$5 bash queue_fas.sh"; }
L(){ setsid bash -c "export PYTHONPATH=\"\${PYTHONPATH:-}\"; $2" \
       > "chain_10k_$1_$H.out" 2>&1 < /dev/null & }

L A "$(for S in 3 4; do Q 0 "0.05 0.10 0.15" 0.0 $S grid_logs_10k_fas_A_$H; done)"
L B "$(for S in 3 4; do Q 1 "0.20 0.25 0.30" 0.0 $S grid_logs_10k_fas_B_$H; done)"
L C "$(for S in 3 4; do Q 0 "0.05 0.10 0.15" 1.0 $S grid_logs_10k_ls_C_$H;  done)"
L D "$(for S in 3 4; do Q 1 "0.20 0.25 0.30" 1.0 $S grid_logs_10k_ls_D_$H;  done)"
```

Each lane does seed 3 fully, then seed 4, so **seed 3 completes first**.

`setsid` (not `nohup`) matters: a killed terminal takes the whole process
group with it otherwise. Verify with `ps -eo pid,ppid,args | grep queue_fas`
-- the chains must show `ppid=1`.

Checkpoint names are `..._alpha_0p20_beta_0p0_..._seed3_...`, so they are
distinct from u24's even inside one `CHECKPOINT_TOP_DIR`. Keeping
`CHECKPOINT_TOP_DIR=/mnt/.../models/$(hostname)` anyway is cheaper than
finding out otherwise.

## Do not edit a launcher while it is running

`run_train_dynamic_grid_fas.sh` is the single biggest trap here. bash reads
a script incrementally and keeps a byte offset across the hours its `python`
line is blocked. Editing the file shifts everything after that offset, and
when training finishes bash resumes mid-token and executes garbage:

```
run_train_dynamic_grid_fas.sh: line 43: above: command not found
```

That happened here and cost two duplicate 4-hour cells. To change a
launcher, copy it to a new filename and point new chains at the copy.

## Memory

The launcher already defaults to `--gpu_memory_fraction 0.46`,
`garbage_collection_threshold:0.7` and `--save_steps 2000`. Two runs fit a
32 GB card with those. See `GPU_MEMORY.md` for why; the short version is
that the caching allocator never gives memory back, so without a cap the
run that meets a large batch first starves its neighbour.

On a 24 GB card two runs do **not** fit (needs ~12.8 GB each including
context). Use one run per GPU, or add `--gradient_checkpointing`, which
costs ~30% speed and changes no result.

## Decoding

Decode on CPU so training never queues behind it. **`watch_decode_cpu.sh`
hardcodes `u24` in its checkpoint glob (line 40) and writes to
`inference_logs_10k/`** -- both need changing before it will see anything
here, and the output dir needs a per-machine name for the same
shared-clone reason as the log dirs:

```bash
H=$(hostname)
sed -e "s|\$N/u24/|\$N/$H/|" -e "s|^OUT=.*|OUT=inference_logs_10k_$H|" \
    watch_decode_cpu.sh > watch_decode_cpu_$H.sh
mkdir -p inference_logs_10k_$H
setsid bash watch_decode_cpu_$H.sh > inference_logs_10k_$H/watch.log 2>&1 &
```

It watches for `checkpoint-10000`, decodes each with greedy (`pipeline`) and
beam=40 on both test splits, and appends to `decode.jsonl`. It skips any
(tag, split, decoder) already in that file, so restarting it is free. `PAR=4`
decodes run at once; `OMP_NUM_THREADS=8` keeps it clear of the four training
dataloaders.

Beam was already CPU-side (lexicon-free flashlight, no LM); only the encoder
forward moves off the GPU. Beam 40 is deliberate -- 20 and 50 differed by
under 0.0001 WER, which is below the floor two machines show decoding the
same checkpoint greedily (0.00006).

## Merging results

`decode.jsonl` lines are self-describing:

```json
{"tag":"FAS_a0.25_seed3","split":"test-clean","decoder":"beam_search","beam":40,"wer":0.0615}
```

The seed is in the tag, so concatenating the two machines' `decode.jsonl`
files is enough -- no row from one can shadow a row from the other. Training-time `eval_wer`
and `pipeline()` greedy do **not** agree, and not by a constant offset, so
any cross-method comparison must use one decoder for every cell.

## Two traps that have already bitten

**One log, two runs.** If a cell restarts, its log gets a second run
appended and a naive parser reading the 16th (or 20th) `eval_wer` picks the
*first* run's final value while the checkpoint on disk is the *second*
run's. Check `grep -c 'auto-generated name' <log>` — it must be 1.

**Same seed, different answer.** Two runs of one config with the same seed
came out 0.0028 and 0.0012 apart, which is at or above the spread across
seeds (SD 0.0016 over four baseline seeds). The dataloader's worker
interleaving is not seeded. Treat a single cell as one sample of a noisy
quantity, not as a fixed number.

**8000-step results are a different experiment.** `max_steps` also sets the
LR decay horizon, so the 8000-step 100hr runs are not 2000 steps short of
these -- they are a different schedule. The step count is in the run name
(`..._100hr_shc_10000steps_...`) to keep them apart; do not pool them.
