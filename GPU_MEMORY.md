# Running two training jobs on one GPU

A 100hr cell is ~4.3 h on a 4090. Two per card nearly doubles throughput,
but three runs died of OOM before the settings below were in place. This is
what went wrong and what fixes it.

## Symptom

```
torch.OutOfMemoryError: Tried to allocate 2.39 GiB.
GPU has 31.36 GiB total, of which 1.50 GiB is free.
This process has 9.56 GiB in use. Process <other> has 20.26 GiB in use.
```

Both OOMs looked identical: the process that died was using only ~9.5 GiB
and asking for ~2.4 GiB more, while its neighbour sat on 19-20 GiB.

## Three numbers, and which one the settings act on

| term | what it is | how to read it |
|---|---|---|
| **allocated** | bytes held by tensors that are alive right now | `torch.cuda.memory_allocated()` |
| **reserved** | bytes the allocator has taken from the driver: allocated **plus** freed blocks it is keeping for reuse | `torch.cuda.memory_reserved()` |
| **nvidia-smi** | reserved plus the CUDA context (~580 MiB) and library workspaces | per-process column |

Reserved only grows when a request cannot be served out of an existing
cached block; then the allocator calls `cudaMalloc` for a new segment. It
does not shrink when tensors are freed. Measured, with a 0.10 cap on a
32109 MiB card:

```
                          allocated  reserved   cached   nvidia-smi
context only                      0         2        2          582
allocate a 1.0 GB tensor        954       962        8         1542
free it                           0       962      962         1542   <- not returned
allocate 0.3 GB                 286       962      676         1542   <- reuses cache
allocate 2.0 GB                1907      1922       15         2502   <- tops up only
empty_cache()                     0         0        0          580
```

So **reserved is a high-water mark**: the most this process has ever needed
at once, plus fragmentation. That is the quantity that ratchets.

`set_per_process_memory_fraction(f)` caps **reserved**, not physical memory.
Asking for 3.73 GiB under a 3211 MiB cap raised OOM while 30.77 GiB of the
card was still free. And `garbage_collection_threshold` measures reserved
against **that cap** -- without a cap its denominator is the whole card, so
it does nothing until 22 GiB.

## Cause

PyTorch's caching allocator never returns memory to the driver. A process's reserved
figure is therefore its own **historical peak**, not what it currently
needs. Two things follow:

- Two runs never have to peak at the same moment to collide. Each one
  ratchets up independently, and the sum of the two high-water marks is
  what has to fit.
- Start times decide the split. A run that starts later expands into
  whatever its neighbour has not claimed yet and keeps it. In the failure
  above the *earlier* run was the one that starved.

Activation memory itself is not the problem. `--max_batch_audio_len`
bounds the padded batch at 6.4M samples = 20,000 frames, and wav2vec2-base
uses SDPA attention, so every term is linear in frames: ~1.2 GiB for the
first conv layer, ~4.8 GiB across the 12 transformer layers, ~1.5 GiB of
weights and AdamW state. About 9 GiB, which matches the 9.5 GiB the dying
process held. The other 10 GiB the neighbour was holding was cache.

## Settings

All three live in `run_train_dynamic_grid_fas.sh` /
`run_train_dynamic_grid_asap.sh`, which are re-read for every cell, so a
change takes effect on each lane's next cell with no restart.

```bash
GPU_MEMORY_FRACTION=0.46     # --gpu_memory_fraction
SAVE_STEPS=6000              # --save_steps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7
```

**`--gpu_memory_fraction 0.46`** calls
`torch.cuda.set_per_process_memory_fraction`, capping each process at
~14.8 GiB of the 31.4 GiB card. No run can take 20 GiB any more. The cap is
~1.6x the ~9 GiB the workload actually needs.

**`garbage_collection_threshold:0.7`** makes the allocator reclaim
least-recently-used cached blocks once reservation passes 70% of that cap,
instead of waiting for an allocation to fail. This is what stops the
ratchet. It needs the cap to be set, since the cap is its denominator.

**`--save_steps 6000`** is the safety net. `save_steps` is otherwise synced
to `max_steps`, so an OOM at step 7000 of 8000 threw away four hours with
nothing to resume from; an intermediate checkpoint makes the worst case
recoverable with `--resume_from_checkpoint`.

It started at 2000 while the cap was unproven. No run has OOM'd since the
cap went in, so the interval is now 6000 -- the midpoint of the 12000-step
schedule. Each checkpoint is about 1.2 GB and writing one stalls both runs
sharing the card, so the tighter interval was paying a real cost for
insurance that has not been claimed.

None of the three changes a number the run produces: two are allocator
policy, one is checkpoint frequency. Cells measured before and after stay
comparable.

## What it does not do

A cap converts "my neighbour starved me" into "this batch does not fit".
If live activations ever exceed 14.8 GiB the run fails deterministically
instead of randomly. That is why `--save_steps` is part of the fix and not
an afterthought.

The only way to remove the risk entirely is one run per GPU, which halves
throughput. Shrinking `--max_batch_audio_len` would also work but changes
which utterances share a batch, so results stop being comparable with
everything measured so far -- and linear LR scaling does not compensate,
because `max_steps` is fixed and a smaller batch means less data seen.

## Checking it is on

```bash
grep '^\[mem\]' <cell>.log          # [mem] capped at 0.460 of 32109 MiB = 14770 MiB
tr '\0' '\n' < /proc/<pid>/environ | grep garbage_collection
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

`nvidia-smi` reports allocator reservation plus ~0.5-1 GiB of CUDA context
per process, so per-process totals run above the cap by that much.
