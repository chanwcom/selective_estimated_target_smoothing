#!/bin/bash
# Usage: bash run_train_dynamic_grid_asap.sh <alpha> <beta> <seed> [profile]
#
# ASAP (Active Support, Acoustic Posterior): FAS's arithmetic with the
# active set read off softmax(logits) instead of off the alignment
# posterior. See apply_active_support_acoustic_smoothing in
# cwk/loss/pytorch/shc_loss_util.py.
#
# ASAP_EPS is NOT interchangeable with FAS_EPS. A softmax has no
# structural zeros, so FAS's 1e-10 marks all C classes active and
# collapses ASAP to textbook uniform LS. Measured mean N_active on a
# trained 32-class model: 1e-2 -> 2.1, 1e-3 -> 3.5, 1e-4 -> 5.7,
# 1e-5 -> 22.2, 1e-6 -> 31.3, 1e-7 -> 32.0.
#
# At a matched eps the two rules select very differently -- at 1e-5 the
# alignment rule keeps 6.8% of classes and the acoustic rule 69.5%, so an
# ASAP cell smooths about 10x harder than the FAS cell of the same alpha.
# Matched eps is not matched strength; read the two arms accordingly.
ALPHA=$1
BETA=$2
SEED=${3:-42}
PROFILE=${4:-libri_light_1hr}

ASAP_EPS=${ASAP_EPS:-1e-5}
MAX_BATCH_AUDIO_LEN=${MAX_BATCH_AUDIO_LEN:-6400000}
MAX_SAMPLE_AUDIO_LEN=${MAX_SAMPLE_AUDIO_LEN:-480000}

# Two runs share a GPU. The caching allocator never hands memory back, so
# each process's reservation is its own historical peak; a run that meets a
# large batch early keeps that share for the rest of its life and the other
# one starves. Both 100hr OOMs looked identical: the dying process held
# 9.5 GiB and asked for 2.4 more while its neighbour sat on 19-20 GiB of a
# 31.4 GiB card. 0.46 caps each side at about 15.0 GiB, which is well above
# the 11.9 GiB either victim actually needed.
#
# This is not a guarantee. A batch whose live activations exceed the cap now
# fails deterministically rather than starving its neighbour, which is why
# SAVE_STEPS exists at all: a failure should cost one checkpoint interval,
# not the whole run. Nothing here changes a number the run produces --
# allocator policy and checkpoint frequency only.
#
# The cap has held since it went in -- no OOM in the runs after it -- so the
# interval is now the midpoint of the 12000-step schedule rather than a
# sixth of it. Each checkpoint is ~1.2 GB and writing one stalls both runs
# sharing the card, so 2000 was buying insurance that is no longer needed.
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.46}
# 7000, not 6000: save_steps must divide max_steps or the run ends with no
# checkpoint at the final step. 6000 into the 14000-step WSD profile saves
# at 6000 and 12000 and nothing at 14000, which is the only checkpoint the
# comparison uses. 7000 gives 7000 and 14000.
# 5000 divides the 15000-step WSD profile, so this lands checkpoints at
# 5000/10000/15000. save_steps MUST divide max_steps: a period that
# does not leaves the run with no checkpoint at the final step, which
# is the only one the comparison uses.
SAVE_STEPS=${SAVE_STEPS:-5000}
# Only for step lists with no usable common period (e.g. 3000,7000,
# 15000). Empty means plain --save_steps; the flag is omitted below
# rather than passed empty, which would swallow the next argument.
SAVE_AT_STEPS=${SAVE_AT_STEPS:-}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF_OVERRIDE:-\
expandable_segments:True,garbage_collection_threshold:0.7}

SAVE_AT_FLAG=""
[ -n "$SAVE_AT_STEPS" ] && SAVE_AT_FLAG="--save_at_steps $SAVE_AT_STEPS"

python wav2vec2_finetuning_sets.py $SAVE_AT_FLAG \
    --alpha=$ALPHA \
    --beta=$BETA \
    --alpha_mode=asap \
    --asap_eps=$ASAP_EPS \
    --smoothing_space=class \
    --vocab_size 32 \
    --finetune_profile=$PROFILE \
    --dynamic_batching \
    --max_batch_audio_len $MAX_BATCH_AUDIO_LEN \
    --max_sample_audio_len $MAX_SAMPLE_AUDIO_LEN \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers \
    --gpu_memory_fraction $GPU_MEMORY_FRACTION \
    --save_steps $SAVE_STEPS \
    --seed $SEED
