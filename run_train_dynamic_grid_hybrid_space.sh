#!/bin/bash
# Usage: bash run_train_dynamic_grid_hybrid_space.sh <alpha> <beta> <seed> [profile]
#
# Corrected L-SETS: same as run_train_dynamic_grid.sh but passes
# --smoothing_space=hybrid, which mixes the target toward
#
#     (1 - beta) * uniform over the C classes
#   + beta       * the label-space masked uniform, scattered to classes
#
# so that
#
#     beta = 0  ->  textbook uniform label smoothing, (1-alpha)*y + alpha/C
#     beta = 1  ->  the historical L-SETS prior, UNCHANGED: blank-heavy
#                   (about half the label positions are blank) and
#                   occurrence-weighted (a label appearing twice in the
#                   transcript collects two shares)
#
# Why only the beta = 0 end moves. The beta = 1 end is a defensible CTC
# prior -- it is blank-heavy because CTC alignments genuinely are, and it
# up-weights repeated labels because several positions can emit them. The
# beta = 0 end was the accident: "uniform" there meant uniform over label
# POSITIONS, which lands in class space as ~49% blank plus an
# occurrence-weighted prior over only the transcript's classes, and zero
# on every other class. beta now interpolates between two methods a reader
# already has a name for.
#
# VERIFIED before launching (cwk/loss/pytorch/shc_loss_hybrid_space_test.py,
# 12 tests):
#
#   beta = 1  ==  label space, to 0.00e+00, end to end through ShcLoss.
#                 Scatter is linear, so blending after the scatter equals
#                 blending before it.
#   beta = 0  ==  (1-alpha)*y + alpha/C, to 0.00e+00.
#   NO PADDING DEPENDENCE AT ANY BETA. One short utterance, run alone vs.
#                 batched behind a 3x longer one, end to end:
#                     hybrid  beta 0.00 .. 1.00   max|grad diff| = 0.0
#                     label   beta 0.00           2.1e-02
#                             beta 0.25/0.50/0.75 1.6e-02/1.1e-02/5.3e-03
#                             beta 1.00           3.0e-08
#                 The old leak scales with (1 - beta) because the component
#                 spread over all L positions is the one weighted by
#                 (1 - beta), and L is the batch's padded width. Replacing
#                 it with 1/C removes the only term that saw the padding.
#   gamma is EXACTLY 0.0 on padded label positions, not merely small, so
#                 the >= eps test excludes them for any eps. This is the
#                 assumption everything above rests on, and it matters
#                 because padded positions carry label id 0 -- blank --
#                 so a leak there would land on the very class most at
#                 risk of being over-weighted.
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
    --smoothing_space=hybrid \
    --vocab_size 32 \
    --finetune_profile=$PROFILE \
    --dynamic_batching \
    --max_batch_audio_len $MAX_BATCH_AUDIO_LEN \
    --max_sample_audio_len $MAX_SAMPLE_AUDIO_LEN \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers \
    --seed $SEED
