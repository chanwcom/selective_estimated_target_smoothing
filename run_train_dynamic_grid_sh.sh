#!/bin/bash
# Usage: bash run_train_dynamic_grid_sh.sh <unused> <seed> [profile]
#
# C-SETS-SH: entropy-matched smoothing with the reference entropy measured
# on the active set only, and NO alpha_max.
#
# The cap is deliberately absent. Every previous entropy-matched run pinned
# its solved alpha at whatever cap it was given (95-99% of logged steps), so
# a capped run is a fixed-alpha run in all but name and would say nothing
# about whether this variant's solve behaves differently. Measured on 1hr
# checkpoints it does: restricting the reference drops the
# trained-alpha -> solved-alpha slope from ~0.95 (an identity map) to ~0.75,
# and makes H(p|A) <= log N hold by construction so the ceiling can no
# longer be unreachable. Capping would hide exactly that.
#
# The first positional is ignored (kept so the single-scalar sweep
# orchestrator can drive this unchanged).
SEED=${2:-42}
PROFILE=${3:-libri_light_1hr}
MAX_BATCH_AUDIO_LEN=${MAX_BATCH_AUDIO_LEN:-6400000}
MAX_SAMPLE_AUDIO_LEN=${MAX_SAMPLE_AUDIO_LEN:-480000}

python wav2vec2_finetuning_sets.py \
    --smoothing_space=class \
    --alpha_mode=entropy_matched_selective \
    --vocab_size 32 \
    --finetune_profile=$PROFILE \
    --dynamic_batching \
    --max_batch_audio_len $MAX_BATCH_AUDIO_LEN \
    --max_sample_audio_len $MAX_SAMPLE_AUDIO_LEN \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers \
    --seed $SEED
