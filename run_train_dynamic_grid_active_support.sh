#!/bin/bash
# Usage: bash run_train_dynamic_grid_active_support.sh <alpha> <beta> <seed> [profile]
#
# Active-support smoothing (AS): textbook uniform smoothing restricted to
# the classes reachable at each frame. Every active class gets alpha/C --
# the same per-class rate uniform smoothing would give -- inactive classes
# get nothing, and only the mass actually handed out (alpha * N_a/C) is
# taken from the target. See apply_active_support_smoothing in
# cwk/loss/pytorch/shc_loss_util.py.
#
# <beta> is accepted and ignored: it exists only so this script keeps the
# `<alpha> <beta> <seed> <profile>` contract run_train_grid_seed.py calls
# with, which is what lets the existing sweep orchestrator, its
# resume-on-restart logic and its summary.json format work unchanged.
#
# The effective smoothing weight is alpha * N_a/C, and N_a is not fixed:
# early in training the alignment posterior is spread over many label
# positions, so most classes are active and this behaves close to ordinary
# uniform smoothing; as the posterior sharpens N_a falls and the smoothing
# anneals itself down. The run logs as_n_active / as_frac / as_alpha_eff
# each logging step so that trajectory is recoverable afterwards.
ALPHA=$1
BETA=$2
SEED=${3:-42}
PROFILE=${4:-libri_light_1hr}

MAX_BATCH_AUDIO_LEN=${MAX_BATCH_AUDIO_LEN:-6400000}
MAX_SAMPLE_AUDIO_LEN=${MAX_SAMPLE_AUDIO_LEN:-480000}

python wav2vec2_finetuning_sets.py \
    --alpha=$ALPHA \
    --beta=0.0 \
    --alpha_mode=active_support \
    --smoothing_space=class \
    --vocab_size 32 \
    --finetune_profile=$PROFILE \
    --dynamic_batching \
    --max_batch_audio_len $MAX_BATCH_AUDIO_LEN \
    --max_sample_audio_len $MAX_SAMPLE_AUDIO_LEN \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers \
    --seed $SEED
