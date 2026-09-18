# u22: RNN-T fine-tuning on the 1 h subset

Companion to `RUN_RNNT_10HR_U22.md`. Separate file because 10 h is
already running there -- this adds the 1 h arm without touching it. u24
is running the 100 h arm.

Everything is identical to the 10 h runbook except the profile and the
schedule, so read that file for the reasoning (memory table, why the
predictor must be the LSTM, why `pkill -f` cannot be used). The
differences are:

| | 10 h (already running) | **1 h (this file)** |
|---|---|---|
| `--finetune_profile` | `libri_light_10hr` | `libri_light_1hr` |
| `--max_steps` | 6000 | **3000** |
| WSD | 1000 / 4000 / 1000 | **1000 / 1500 / 500** |
| `--eval_steps` | 2000 | **1000** |
| wall clock per cell | ~1.1 h | **~0.5 h** |

The schedule is the 1 h profile's own WSD after the recent upgrade, i.e.
the same one the final CTC 1 h grid used, so the transducer numbers sit
against a CTC baseline trained on the same schedule. The three parts must
sum to `--max_steps` or the script asserts.

## 0. Preconditions

Already satisfied if 10 h is running. If you are starting fresh:

```bash
cd /mnt/synology_nas_00/chanwcom/local_repository/selective_estimated_target_smoothing && git pull
cd ../cognitive_workflow_kit && git pull && cd -
source ./set_config.sh
cd ../cognitive_workflow_kit/cwk/loss/pytorch && python rnnt_shc_loss_test.py && cd -
```

That must print 12 tests and `OK`.

## 1. Grid

Same as 10 h and as the CTC grid: baseline plus FAS and LS at
alpha = 0.05 0.10 0.15 0.20 0.25 0.30, seeds 0..4 in order.
13 cells per seed, 65 total, about 33 GPU-hours.

## 2. Run it

**Use a different working directory from the 10 h run.** The claim
directory is what stops two workers taking the same cell, and mixing the
two grids in one directory would let a 1 h worker claim a 10 h tag.

```bash
cd /mnt/synology_nas_00/chanwcom/local_repository/selective_estimated_target_smoothing
W=$HOME/rnnt1hr                       # note: NOT rnnt10hr
mkdir -p $W/claim $W/logs $W/ckpt

python3 - <<'PY'
import os
W=os.path.expanduser("~/rnnt1hr")
L=[]
for seed in range(5):
    L.append(f"base 0.0 0.0 {seed}")
    for a in (0.05,0.10,0.15,0.20,0.25,0.30): L.append(f"FAS {a} 0.0 {seed}")
    for a in (0.05,0.10,0.15,0.20,0.25,0.30): L.append(f"LS {a} 1.0 {seed}")
open(f"{W}/queue.txt","w").write("\n".join(L)+"\n")
print(len(L),"cells")
PY

cat > $W/worker.sh <<'EOF2'
#!/bin/bash
set -u
cd /mnt/synology_nas_00/chanwcom/local_repository/selective_estimated_target_smoothing
export PYTHONPATH="${PYTHONPATH:-}"; source ./set_config.sh >/dev/null 2>&1
export PATH="$PYTHON_BIN:$PATH"
W=$1; GPU=$2
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
while read -r METH ALPHA BETA SEED; do
  TAG="1hr_${METH}_a${ALPHA}_s${SEED}"
  mkdir "$W/claim/$TAG" 2>/dev/null || continue
  MODE=fixed; [ "$METH" != "base" ] && MODE=floored_active_support
  echo "=== $(date '+%m-%d %H:%M') gpu$GPU start $TAG"
  nice -n 19 python wav2vec2_rnnt.py \
    --finetune_profile libri_light_1hr \
    --max_steps 3000 --warmup_steps 1000 \
    --num_stable_steps 1500 --num_decay_steps 500 \
    --dynamic_batching --max_batch_audio_len 3200000 --grad_accum 2 \
    --max_sample_audio_len 480000 --max_label_len 450 \
    --dataloader_num_workers 2 --encoder_stride 2 --predictor lstm \
    --alpha $ALPHA --beta $BETA --alpha_mode $MODE --seed $SEED \
    --eval_steps 1000 --eval_examples 200 --log_every 500 \
    --output_dir $W/ckpt/$TAG \
    > $W/logs/$TAG.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') gpu$GPU done $TAG rc=$?"
done < $W/queue.txt
echo "=== gpu$GPU queue drained $(date '+%m-%d %H:%M')"
EOF2
chmod +x $W/worker.sh
```

Then start one worker per GPU you want to give it. **Do not start a
worker on a GPU that is already running a 10 h cell** -- the worker
claims a cell immediately, so a second cell would land on the same card
and both would OOM (10.86 GiB each at this budget, 23 GiB available).

```bash
nohup nice -n 19 bash $W/worker.sh $W 2 > $W/worker2.log 2>&1 &   # pick free GPUs
nohup nice -n 19 bash $W/worker.sh $W 3 > $W/worker3.log 2>&1 &
```

If every GPU is busy with 10 h, wait for a 10 h cell to finish and start
the 1 h worker on that card then. Workers can be added at any time; the
`mkdir` claim makes that safe.

## 3. Watch it

```bash
W=$HOME/rnnt1hr
ls $W/claim | wc -l                             # cells started (of 65)
grep -ch done $W/worker*.log                    # cells finished
grep -h 'dev-clean=' $W/logs/*.log | tail -20
grep -h peak $W/logs/*.log | sort -u | tail -5   # must stay near 10.9 GiB
grep -h dropped $W/logs/*.log | grep -v 'dropped=0' | head   # should be empty
```

A healthy cell:

```
[500/3000] loss=... B=13 T=376 U=255 peak=10.86GiB dropped=0 0.6s/step
[1000] dev-clean=0.2xxxx(n=200)  dev-other=0.4xxxx(n=200)  e.g. 'THE ...'
[done] steps=3000 peak=10.86GiB dropped=0 wall=30.0min
```

1 h is a small subset, so an epoch is only a handful of batches and the
`[epoch N] M batches` lines scroll fast -- that is expected, not a bug.
What matters is that the loss falls and both dev numbers fall with it.

For reference, u24 measured a single-GPU 1 h transducer run reaching
dev-clean 0.217 at 2000 steps on a 4x smaller budget, so a 3000-step run
at this budget should land below that. If a cell ends with dev numbers
near 1.0 and empty hypotheses, the predictor is wrong -- check that
`--predictor lstm` is on the command line.

## 4. Stopping and requeueing

Same as the 10 h runbook: `kill` the worker shell by PID to stop it
taking new cells (the running cell finishes), then kill the cell's python
by PID if you want it gone too. Never `pkill -f` -- the pattern matches
your own shell. A cell that dies leaves `$W/claim/<TAG>`; `rmdir` just
that directory to put it back in the queue.

## 5. Sending results back

`$W/ckpt` is local to u22. When the grid finishes, either rsync the
checkpoints to `/mnt/synology_nas_00/chanwcom/models/<host>/` as the CTC
runs do, or send `$W/logs/*.log` -- they carry the eval curve, peak
memory and wall time, which is enough to confirm the runs are healthy
before the full-split decode that produces the paper numbers.
