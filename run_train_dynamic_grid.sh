#!/bin/bash
# Usage: bash run_train_dynamic_grid.sh <alpha> <beta> <seed> [profile]
#
# One cell of the alpha x beta grid sweep (see run_train_grid_seed.py). Dynamic
# batching. `profile` is the 4th positional arg (any --finetune_profile
# choice in wav2vec2_finetuning_sets.py, e.g. libri_light_1hr,
# libri_light_10hr, libri_speech_clean_100hr, or a future gigaspeech_xs
# profile once that dataset is prepared) -- defaults to libri_light_1hr
# if omitted, so existing 1hr sweeps/commands keep working unchanged.
#
# MAX_BATCH_AUDIO_LEN / MAX_SAMPLE_AUDIO_LEN can be overridden via
# environment variables (e.g. `MAX_BATCH_AUDIO_LEN=9000000 bash
# run_train_dynamic_grid.sh ...`) without editing this file -- useful once
# a dataset with a meaningfully different length distribution (e.g.
# gigaspeech, which may skew longer/shorter than LibriSpeech) needs its
# own budget. Defaults below are calibrated specifically for LibriSpeech
# on this 4090 -- see the values' own comments for how -- and are NOT
# guaranteed to still make sense for a dataset with a different length
# distribution; re-profile before trusting them on anything but
# LibriSpeech-like data.
ALPHA=$1
BETA=$2
SEED=${3:-42}
PROFILE=${4:-libri_light_1hr}

# --max_batch_audio_len default (6,400,000): sized to this 4090's measured
# throughput-saturation point (batch~24 -- beyond that, more batch = no
# more samples/sec, just more OOM risk, confirmed by direct profiling)
# against the real LibriSpeech length distribution (mean=12.3s, p99=16.7s,
# max=29.73s across the full 960h train set, scanned 2026-08-18).
MAX_BATCH_AUDIO_LEN=${MAX_BATCH_AUDIO_LEN:-6400000}

# --max_sample_audio_len default (480000 = 30s): generous safety cap --
# real LibriSpeech max observed was 29.73s, so this only excludes truly
# pathological outliers, not real data.
MAX_SAMPLE_AUDIO_LEN=${MAX_SAMPLE_AUDIO_LEN:-480000}

python wav2vec2_finetuning_sets.py \
    --alpha=$ALPHA \
    --beta=$BETA \
    --vocab_size 32 \
    --finetune_profile=$PROFILE \
    --dynamic_batching \
    --max_batch_audio_len $MAX_BATCH_AUDIO_LEN \
    --max_sample_audio_len $MAX_SAMPLE_AUDIO_LEN \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers \
    --seed $SEED
