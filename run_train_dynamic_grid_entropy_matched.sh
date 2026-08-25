#!/bin/bash
# Usage: bash run_train_dynamic_grid_entropy_matched.sh <alpha_max> <seed> [profile]
#
# One cell of the C-SETS-H sweep: SETS in class space with the smoothing
# weight SOLVED FOR per example rather than tuned, so that the smoothed
# target's mean per-frame entropy equals the model's own acoustic
# posterior's. See cwk/loss/pytorch/shc_loss_util.py's
# apply_entropy_matched_smoothing for the derivation.
#
# NOTE ON THE FIRST ARGUMENT. It is --entropy_match_alpha_max, NOT --alpha.
# --alpha/--beta are ignored entirely in this mode. It is passed as the
# first positional so that run_train_grid_seed_peak_capping.py (whose
# sweep axis is a single scalar) can drive this sweep unchanged; that
# orchestrator writes summary.json keys spelled "alpha=<value>", which
# here should be read as "alpha_max=<value>".
#
# alpha_max=1.0 is the headline configuration: no clamp at all, the
# entropy-matching equation alone decides. Smaller values are a safety
# net for the case where the solved alpha turns out far larger than the
# tuned optimum (0.02 at 1hr, 0.01 at 10hr) -- which is the main open
# risk this sweep exists to measure.
#
# MAX_BATCH_AUDIO_LEN / MAX_SAMPLE_AUDIO_LEN can be overridden via
# environment variables, same as run_train_dynamic_grid.sh -- see that
# file's comments for the calibration behind the defaults below.
ALPHA_MAX=$1
SEED=${2:-42}
PROFILE=${3:-libri_light_1hr}

MAX_BATCH_AUDIO_LEN=${MAX_BATCH_AUDIO_LEN:-6400000}
MAX_SAMPLE_AUDIO_LEN=${MAX_SAMPLE_AUDIO_LEN:-480000}

python wav2vec2_finetuning_sets.py \
    --smoothing_space=class \
    --alpha_mode=entropy_matched \
    --entropy_match_alpha_max=$ALPHA_MAX \
    --vocab_size 32 \
    --finetune_profile=$PROFILE \
    --dynamic_batching \
    --max_batch_audio_len $MAX_BATCH_AUDIO_LEN \
    --max_sample_audio_len $MAX_SAMPLE_AUDIO_LEN \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers \
    --seed $SEED
