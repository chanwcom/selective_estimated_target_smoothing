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

python wav2vec2_finetuning_sets.py \
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
    --seed $SEED
