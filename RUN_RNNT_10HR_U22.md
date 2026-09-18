# u22: RNN-T fine-tuning on the 10 h subset

u24 is running the same experiment on 1 h and 100 h. This file is the 10 h
arm. Both machines share the NAS, so the code is already identical -- do
not edit any source file, only run what is below.

## 0. Pull first

```bash
cd /mnt/synology_nas_00/chanwcom/local_repository/selective_estimated_target_smoothing
git pull
cd ../cognitive_workflow_kit && git pull && cd -
```

You need at least these:

* `cognitive_workflow_kit` @ `7654078` -- `cwk/loss/pytorch/rnnt_shc_loss.py`,
  the transducer counterpart of `ShcLoss`.
* `selective_estimated_target_smoothing` @ `8389120` -- `wav2vec2_rnnt.py`.

Sanity check the loss before spending GPU time on it (about 20 s, CPU only):

```bash
source ./set_config.sh
cd ../cognitive_workflow_kit/cwk/loss/pytorch
python rnnt_shc_loss_test.py     # must be 12 tests, OK
cd -
```

If that is not `OK`, stop and tell me -- everything below is meaningless
otherwise.

## 1. What this is

Same three methods as the CTC grid, same alpha grid, same seeds:

| | |
|---|---|
| methods | baseline (alpha=0), FAS (`--alpha_mode floored_active_support --beta 0.0`), LS (same mode, `--beta 1.0`) |
| alpha | 0.05 0.10 0.15 0.20 0.25 0.30 |
| seeds | 0 1 2 3 4, in that order |
| cells | 1 + 6 + 6 = 13 per seed, 65 total |

The transducer differs from the CTC runs in exactly two places, both
deliberate:

* **Frame rate.** `--encoder_stride 2`, i.e. 40 ms frames. That is the
  transducer convention (NeMo's Conformer-Transducer, Emformer-RNNT,
  icefall all sit at 25-33 fps) where CTC conventionally runs at
  wav2vec2's native 20 ms. Absolute WERs are therefore NOT comparable
  across the two losses; every comparison we draw is within one loss,
  where the frame rate is fixed.
* **Predictor.** A single-layer LSTM, trained from scratch (there is no
  pre-trained transducer predictor). Do not switch it to
  `--predictor stateless`: with the default 2-label context a
  single-batch overfit test plateaus at loss ~100 and emits only the
  first two words, because the same 2-label context recurs many times
  within one transcript with different continuations.

`--max_label_len 450` bounds the joint tensor's U axis. On LibriSpeech it
drops nothing (measured max U = 401), but without it peak memory is an
observation rather than a guarantee.

## 2. Memory -- read this before choosing the GPU layout

Measured peak on 100 h at `--encoder_stride 2`, worst batch taken from a
full-epoch scan of batch shapes:

| `--max_batch_audio_len` | peak | batch (mean utts) |
|---|---|---|
| 1600000 (100 s) | 5.80 GiB | 7 |
| 3200000 (200 s) | 10.86 GiB | 14 |
| 6400000 (400 s) | 20.68 GiB | 27 |

6400000 is the CTC runs' budget, so it keeps the optimizer's batch
identical and is the default in the script.

**On u22's 24 GB 4090s that leaves only about 2.3 GiB of headroom.** It
will probably run, but it is not something to leave unattended for days --
one fragmentation spike and the cell dies mid-run. Use this instead:

```
--max_batch_audio_len 3200000 --grad_accum 2
```

Same effective batch (400 s per optimizer step), 10.86 GiB peak, so one
job per card with 12 GiB of headroom, or two jobs per card at 21.7 GiB if
you are willing to sit at 1.3 GiB of headroom (I would not).

**One job per GPU.** Do not set `GPU_MEMORY_FRACTION`; `wav2vec2_rnnt.py`
does not read it.

## 3. Run it

Copy the queue and worker below verbatim. `mkdir` is atomic, so workers
can be added or removed at any time and never take the same cell twice --
start one per GPU you are willing to give it, and add another later by
running the same command with a different GPU index.

```bash
cd /mnt/synology_nas_00/chanwcom/local_repository/selective_estimated_target_smoothing
W=$HOME/rnnt10hr                      # LOCAL, not on the NAS: the claim
mkdir -p $W/claim $W/logs $W/ckpt     # directory must not be shared with u24

python3 - <<'PY'
import os
W=os.path.expanduser("~/rnnt10hr")
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
  TAG="10hr_${METH}_a${ALPHA}_s${SEED}"
  mkdir "$W/claim/$TAG" 2>/dev/null || continue
  MODE=fixed; [ "$METH" != "base" ] && MODE=floored_active_support
  echo "=== $(date '+%m-%d %H:%M') gpu$GPU start $TAG"
  nice -n 19 python wav2vec2_rnnt.py \
    --finetune_profile libri_light_10hr \
    --max_steps 6000 --warmup_steps 1000 \
    --num_stable_steps 4000 --num_decay_steps 1000 \
    --dynamic_batching --max_batch_audio_len 3200000 --grad_accum 2 \
    --max_sample_audio_len 480000 --max_label_len 450 \
    --dataloader_num_workers 2 --encoder_stride 2 --predictor lstm \
    --alpha $ALPHA --beta $BETA --alpha_mode $MODE --seed $SEED \
    --eval_steps 2000 --eval_examples 200 --log_every 500 \
    --output_dir $W/ckpt/$TAG \
    > $W/logs/$TAG.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') gpu$GPU done $TAG rc=$?"
done < $W/queue.txt
echo "=== gpu$GPU queue drained $(date '+%m-%d %H:%M')"
EOF2
chmod +x $W/worker.sh

nohup nice -n 19 bash $W/worker.sh $W 0 > $W/worker0.log 2>&1 &
nohup nice -n 19 bash $W/worker.sh $W 1 > $W/worker1.log 2>&1 &
```

The schedule is the 10 h profile's own WSD, taken from
`_FINETUNE_PROFILES["libri_light_10hr"]` after the recent upgrade:
1000 warmup + 4000 stable + 1000 decay = 6000 steps. The three must sum to
`--max_steps` or the script asserts. (1 h is 1000/1500/500 = 3000 and 100 h
is 1000/10000/4000 = 15000; do not mix them up.)

## 4. Watch it

```bash
W=$HOME/rnnt10hr
ls $W/claim | wc -l                                # cells started
grep -c done $W/worker*.log                        # cells finished
tail -3 $W/logs/$(ls -t $W/logs | head -1)         # newest cell
grep -h 'dev-clean=' $W/logs/*.log | tail -20      # eval so far
grep -h peak $W/logs/*.log | sort -u | tail -5     # peak memory actually seen
```

A healthy cell looks like this (`peak` must stay well under 23 GiB):

```
[500/6000] loss=... B=13 T=376 U=255 peak=10.86GiB dropped=0 0.6s/step
[2000] dev-clean=0.2xxxx(n=200)  dev-other=0.4xxxx(n=200)  e.g. 'THE ...'
```

`dropped=0` means `--max_label_len` threw nothing away, which is what to
expect on LibriSpeech. If `dropped` climbs, tell me -- the joint tensor's
U axis is then being clipped and the numbers are not comparable with u24's.

Periodic eval is greedy decoding of 200 utterances per split, which is for
watching the curve only. The numbers that go in the paper come from a
separate full-split decode afterwards; do not read the 200-utterance
values as results.

## 5. Stopping

Kill order matters and pattern matching does not work here -- `pkill -f`
matches your own shell's command line and kills the terminal instead.
Always go parent first, then child, by PID:

```bash
ps -eo pid,args | grep '[w]orker.sh'          # worker PIDs
kill -TERM <worker pid>                        # stops it taking new cells
ps -eo pid,args | grep '[w]av2vec2_rnnt.py'   # the running cell
kill -TERM <python pid>                        # stops the cell itself
```

Killing only the worker leaves the current cell running to completion,
which is usually what you want. A cell that dies leaves its
`$W/claim/<TAG>` directory behind and will NOT be retried -- `rmdir` that
one directory to requeue it.

## 6. What to send back

Nothing needs copying by hand; the checkpoints under `$W/ckpt` are local
to u22, so when the runs finish, either
`rsync` them to `/mnt/synology_nas_00/chanwcom/models/<host>/` the way the
CTC runs do, or just send `$W/logs/*.log` -- the logs carry the eval
curve, the peak memory and the wall time, which is enough to check the
runs are healthy before the full-split decode.
