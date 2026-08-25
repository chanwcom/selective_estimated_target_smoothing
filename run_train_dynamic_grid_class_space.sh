#!/bin/bash
# Usage: bash run_train_dynamic_grid_class_space.sh <alpha> <beta> <seed> [profile]
#
# Class-space counterpart of run_train_dynamic_grid.sh: identical in every
# respect except it passes --smoothing_space=class, so the SETS smoothing is
# applied AFTER the L -> C scatter (over actual output classes) instead of
# before it (over blank-augmented label positions).
#
# Why this exists: every SETS / PP-SETS / PC-SETS result so far was produced
# in label space, where a "uniform" mixin is not uniform over classes at all
# -- measured at roughly 49% blank, with the remaining mass spread over only
# the classes that occur in that utterance's transcript, weighted by how
# often they occur, and exactly 0 on every other class. That is much closer
# to unigram label smoothing with a strong CTC blank prior than to classical
# uniform label smoothing. This script runs the genuinely-uniform-over-
# classes version so the two can be compared directly, which also removes
# the space choice as a confound for any later class-space method (e.g.
# entropy-matched smoothing, which has to live in class space to be able to
# compare its target's entropy against the acoustic posterior's).
#
# See cwk/loss/pytorch/shc_loss_util.py's module docstring for the full
# derivation and the measured numbers.
#
# MAX_BATCH_AUDIO_LEN / MAX_SAMPLE_AUDIO_LEN can be overridden via
# environment variables, same as run_train_dynamic_grid.sh -- see that
# file's comments for the calibration behind the defaults below.
ALPHA=$1
BETA=$2
SEED=${3:-42}
PROFILE=${4:-libri_light_1hr}

MAX_BATCH_AUDIO_LEN=${MAX_BATCH_AUDIO_LEN:-6400000}
MAX_SAMPLE_AUDIO_LEN=${MAX_SAMPLE_AUDIO_LEN:-480000}

python wav2vec2_finetuning_sets.py \
    --alpha=$ALPHA \
    --beta=$BETA \
    --smoothing_space=class \
    --vocab_size 32 \
    --finetune_profile=$PROFILE \
    --dynamic_batching \
    --max_batch_audio_len $MAX_BATCH_AUDIO_LEN \
    --max_sample_audio_len $MAX_SAMPLE_AUDIO_LEN \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers \
    --seed $SEED
