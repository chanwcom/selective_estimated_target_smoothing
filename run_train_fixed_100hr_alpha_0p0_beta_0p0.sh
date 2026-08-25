##PROFILE=libri_speech_full_960hr
PROFILE=libri_speech_clean_100hr
##PROFILE=libri_light_10hr
#PROFILE=libri_light_1hr
#
#python wav2vec2_finetuning_sets.py \
#    --alpha=0.0  \
#    --beta=0.0   \
#    --vocab_size 32 \
#    --finetune_profile=$PROFILE  \
#    --dynamic_batching \
#    --max_batch_audio_len 10000000 \
#    --dataloader_num_workers 4 \
#    --dataloader_persistent_workers
#    # --per_device_train_batch_size 24  \
#    # --length_bucket_window_mult 0
#    #
#

# 25 seconds cut
# LibriSpeech train set (train-clean-100 + train-clean-360 + train-other-500),
# scanned 2026-08-18: 281,241 utterances, 961.1 hours total.
# mean length = 12.3s; only 9 utterances (0.003%) exceed 25s; max = 29.73s.
#
# Optional first positional arg = seed (default 42). run_multi_seed.py
# calls this as `bash run_train_fixed_100hr_alpha_0p0_beta_0p0.sh <seed>` for
# multi-seed comparisons; --seed also feeds the auto-generated --run_name,
# so repeats with different seeds land in separate checkpoint dirs instead
# of overwriting each other.
SEED=${1:-42}
python wav2vec2_finetuning_sets.py \
--alpha=0.0  \
--beta=0.0   \
--vocab_size 32 \
--finetune_profile=$PROFILE  \
--per_device_train_batch_size 24 \
--length_bucket_window_mult 0 \
--max_sample_audio_len 400000 \
--dataloader_num_workers 4 \
--dataloader_persistent_workers \
--seed $SEED
# --dynamic_batching \
# --max_batch_audio_len 10000000
