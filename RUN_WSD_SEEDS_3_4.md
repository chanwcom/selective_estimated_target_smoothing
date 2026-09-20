# Seeds 3 and 4 of the WSD 100hr grid, on the 4090 machine

u24 is running seeds **0, 1, 2**. This machine runs seeds **3 and 4** of the
same grid, from the same scripts in this same NAS clone.

## The grid

| | |
|---|---|
| profile | `libri_speech_clean_100hr_wsd` |
| schedule | WSD: warmup 0→1000, **constant 1000→11000**, linear decay 11000→**15000** |
| peak LR | 1e-4 (`--gpu_profile 4090`, the default on both machines) |
| checkpoints | every 5000 → 5000 / 10000 / **15000** |
| eval | every 1000 |
| FAS | `--beta 0.0`, `--fas_eps 1e-10` |
| textbook LS | `--beta 1.0` (bit-identical to uniform LS) |
| alpha | 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 |
| seeds here | **3, 4** |
| no baseline | α=0 is not part of this grid |

6 alphas × 2 methods × 2 seeds = **24 cells**.

## Before launching

```bash
cd <this repo>
git pull                          # only if this checkout is NOT the NAS one
grep -A14 'libri_speech_clean_100hr_wsd' wav2vec2_finetuning_sets.py \
  | grep -E 'max_steps|warmup_steps|eval_steps|num_stable|num_decay|decay_type'
#   -> max_steps=15000, warmup_steps=1000, eval_steps=1000,
#      num_stable_steps=10000, num_decay_steps=4000, decay_type="linear"
grep 'CHECKPOINT_TOP_DIR' config.local.sh
#   -> this machine's own path; /mnt/.../models/$(hostname) keeps it on the
#      NAS without colliding with u24
```

`config.local.sh` is gitignored, so each machine sets its own.

Note the profile name: **`libri_speech_clean_100hr_wsd`**, not
`libri_speech_clean_100hr`. The latter was the old 12000-step linear-decay
profile; passing it produced a run that looked fine and was comparable to
nothing here, so on 2026-09-20 it was removed outright. That name is now an
invalid `--finetune_profile` choice and fails before training starts.

## One run per GPU on 24 GB

**`GPU_MEMORY_FRACTION=0` is not optional here.** The launcher defaults to
0.46, which was chosen for a 32 GB card carrying two runs (0.46 × 31.4 GiB
= 14.4 GiB each). On a 24 GB card 0.46 is 11.0 GiB — *below* the 11.9 GiB a
run actually needs — so leaving the default turns a run that would have fit
into a deterministic OOM. 0 disables the cap, which is what a GPU with a
single tenant wants: there is no neighbour to starve.

Two runs do **not** fit on 24 GB (~12.8 GiB each including CUDA context).
One run per GPU, so two lanes on a two-GPU box, twelve cells per lane.

## Launch

```bash
cd <this repo>
H=$(hostname)
export GPU_MEMORY_FRACTION=0          # REQUIRED on 24 GB — see above
P=libri_speech_clean_100hr_wsd
Q(){ echo "GPU=$1 ALPHAS=\"$2\" BETAS=\"$3\" SEEDS=\"$4\" FAS_EPS=1e-10 \
PROFILE=$P LOG_DIR=$5 bash queue_fas.sh"; }
L(){ setsid bash -c "export PYTHONPATH=\"\${PYTHONPATH:-}\"; $2" \
       > "chain_wsd_$1_$H.out" 2>&1 < /dev/null & }

A="0.05 0.10 0.15 0.20 0.25 0.30"
L fas "$(for S in 3 4; do Q 0 "$A" 0.0 $S grid_logs_wsd_fas_$H; done)"
L ls  "$(for S in 3 4; do Q 1 "$A" 1.0 $S grid_logs_wsd_ls_$H;  done)"
```

FAS on GPU 0, LS on GPU 1, six alphas each, seed 3 fully and then seed 4 —
so **seed 3 completes first**.

`setsid` (not `nohup`) matters: a killed terminal takes the whole process
group with it otherwise. Verify with `ps -eo pid,ppid,args | grep queue_fas`
— the chains must show `ppid=1`.

**Every path is suffixed with `$(hostname)` on purpose.** This repo is a
single clone on the NAS and both machines run out of it, so a bare
`grid_logs_wsd_fas_A` would be the *same directory* on both;
`run_train_grid_seed.py` read-modify-writes `summary.json` in it and the two
machines would clobber each other. Chain stdout files collide the same way.

Checkpoint directory names carry `seed3`/`seed4`, so they are distinct from
u24's even inside one `CHECKPOINT_TOP_DIR`. Keeping
`CHECKPOINT_TOP_DIR=/mnt/.../models/$(hostname)` anyway is cheaper than
finding out otherwise.

## Stopping a lane without killing its training

Order matters, top down: **chain → runner → launcher → python.**

- Killing the *runner* (`run_train_grid_seed.py`) **also kills the cell it
  is running** — the child does not survive.
- Killing a *chain* first stops it from starting the next seed. Killing the
  runner first makes the chain see its child exit and immediately launch
  the next command in the list.

Both of those were learned the hard way here: five cells at 75–90%
completion were lost to the wrong order.

To stop everything cleanly:

```bash
pkill -TERM -f 'bash -c export PYTHONPATH'   # chains
sleep 4
pkill -TERM -f run_train_grid_seed.py        # runners (takes cells with it)
sleep 4
pkill -TERM -f run_train_dynamic_grid        # launchers
sleep 3
pkill -TERM -f wav2vec2_finetuning_sets.py   # training
```

## Never edit a launcher while it is running

`run_train_dynamic_grid_fas.sh` is the biggest trap. bash reads a script
incrementally and keeps a byte offset across the hours its `python` line is
blocked. Editing in place shifts everything after that offset, and when
training finishes bash resumes mid-token:

```
run_train_dynamic_grid_fas.sh: line 43: above: command not found
```

That cost two duplicate 4-hour cells. If you must change one, write a new
file and `mv` it over the old name — a rename gives a new inode, the
running bash keeps reading the old one to the end, and the next cell opens
the new one. In-place editing (`sed -i` on the same inode, an editor's
save) is what breaks it.

## Decoding

Decode on CPU so training never queues behind it. `watch_decode_cpu.sh`
hardcodes `u24` in its glob and the 12000-step checkpoint name, so generate
a per-machine copy:

```bash
H=$(hostname)
sed -e "s|\$N/u24/|\$N/$H/|" \
    -e "s|\*100hr_shc_12000steps\*/checkpoint-12000|*100hr_wsd_shc_15000steps*/checkpoint-15000|" \
    -e "s|^OUT=.*|OUT=inference_logs_wsd_$H|" \
    watch_decode_cpu.sh > watch_decode_cpu_$H.sh
mkdir -p inference_logs_wsd_$H
setsid bash watch_decode_cpu_$H.sh > inference_logs_wsd_$H/watch.log 2>&1 &
```

It decodes each `checkpoint-15000` with greedy (`pipeline`) and beam=40 on
test-clean and test-other, four jobs in parallel, and appends to
`decode.jsonl`. It skips any (tag, split, decoder) already recorded, so
restarting it is free. Beam is CPU-side anyway (lexicon-free flashlight, no
LM); only the encoder forward moves off the GPU.

Beam 40 is deliberate: 20 and 50 differed by under 0.0001 WER, below the
floor two machines show decoding the same checkpoint greedily (0.00006).

## Merging results

`decode.jsonl` lines are self-describing and the seed is in the tag, so
concatenating both machines' files is enough:

```json
{"tag":"FAS_a0.2_seed3","split":"test-clean","decoder":"beam_search","beam":40,"wer":0.0599}
```

Training-time `eval_wer` and `pipeline()` greedy do **not** agree, and not
by a constant offset — measured here, cells swap rank between the two. Any
cross-method comparison must use one decoder for every cell.

## Three traps already hit here

**One log, two runs.** If a cell restarts, its log gets a second run
appended and a parser reading the Nth `eval_wer` picks the *first* run's
value while the checkpoint on disk is the *second* run's. Check
`grep -c 'auto-generated name' <log>` — it must be 1.

**Same seed, different answer.** Two runs of one config at the same seed
came out 0.0028 and 0.0012 apart, at or above the spread across seeds. The
dataloader's worker interleaving is not seeded. A single cell is one sample
of a noisy quantity, not a fixed number.

**Different max_steps is a different experiment.** `max_steps` sets the
decay horizon, so the 8000-, 12000- and 15000-step grids are three
schedules, not longer versions of one. The step count and the profile name
are both in the run name to keep them apart; never pool them in one table.
