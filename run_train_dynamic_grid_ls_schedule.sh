#!/bin/bash
# Usage: ALPHA_AFTER=<a> ALPHA_SWITCH_STEP=<n> \
#            bash run_train_dynamic_grid_ls_schedule.sh <alpha> <beta> <seed> [profile]
#
# Classical uniform label smoothing with a piecewise-constant schedule:
# --alpha for steps below ALPHA_SWITCH_STEP, ALPHA_AFTER from there on.
#
# Same class-space setup as run_train_dynamic_grid_class_space.sh -- with
# beta=0.0 that is exactly textbook (1-alpha)*y + alpha/C smoothing, see
# queue_uniform_ls.sh for why label space is NOT that. The only thing added
# here is the time axis: whether smoothing helps early (as a warmup
# regularizer, annealed away before the model commits) or late (as a
# calibrator, once the CTC alignment has already sharpened) is not
# answerable from any fixed-alpha sweep, since both collapse to the same
# constant.
#
# The schedule is passed via environment rather than positionals because
# run_train_grid_seed.py's contract is fixed at `<alpha> <beta> <seed>
# <profile>`; keeping that contract lets the existing sweep orchestrator,
# its resume-on-restart logic and its summary.json format work unchanged.
ALPHA=$1
BETA=$2
SEED=${3:-42}
PROFILE=${4:-libri_light_1hr}

ALPHA_AFTER=${ALPHA_AFTER:?set ALPHA_AFTER (smoothing weight after the switch)}
ALPHA_SWITCH_STEP=${ALPHA_SWITCH_STEP:?set ALPHA_SWITCH_STEP (step to switch at)}

MAX_BATCH_AUDIO_LEN=${MAX_BATCH_AUDIO_LEN:-6400000}
MAX_SAMPLE_AUDIO_LEN=${MAX_SAMPLE_AUDIO_LEN:-480000}

python wav2vec2_finetuning_sets.py \
    --alpha=$ALPHA \
    --beta=$BETA \
    --alpha_switch_step=$ALPHA_SWITCH_STEP \
    --alpha_after_switch=$ALPHA_AFTER \
    --smoothing_space=class \
    --vocab_size 32 \
    --finetune_profile=$PROFILE \
    --dynamic_batching \
    --max_batch_audio_len $MAX_BATCH_AUDIO_LEN \
    --max_sample_audio_len $MAX_SAMPLE_AUDIO_LEN \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers \
    --seed $SEED
