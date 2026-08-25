#!/bin/bash
# Usage: bash run_train_dynamic_grid_peak_capping.sh <alpha> <seed> [profile]
#
# One cell of an alpha sweep for the peak-capping SETS variant (see
# run_train_grid_seed_peak_capping.py). Sibling of
# run_train_dynamic_grid_peak_preserving.sh (same dynamic-batching setup),
# but sweeps --alpha with --peak_capping instead of --gamma with
# --peak_preserving. `profile` is the 3rd positional arg -- defaults to
# libri_light_1hr if omitted.
#
# MAX_BATCH_AUDIO_LEN / MAX_SAMPLE_AUDIO_LEN can be overridden via
# environment variables, same as run_train_dynamic_grid.sh -- see that
# file's comments for the calibration behind the defaults below.
ALPHA=$1
SEED=${2:-42}
PROFILE=${3:-libri_light_1hr}

MAX_BATCH_AUDIO_LEN=${MAX_BATCH_AUDIO_LEN:-6400000}
MAX_SAMPLE_AUDIO_LEN=${MAX_SAMPLE_AUDIO_LEN:-480000}

python wav2vec2_finetuning_sets.py \
    --peak_capping \
    --alpha=$ALPHA \
    --vocab_size 32 \
    --finetune_profile=$PROFILE \
    --dynamic_batching \
    --max_batch_audio_len $MAX_BATCH_AUDIO_LEN \
    --max_sample_audio_len $MAX_SAMPLE_AUDIO_LEN \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers \
    --seed $SEED
