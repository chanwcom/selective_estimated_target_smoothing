PROFILE=libri_speech_full_960hr
#PROFILE=libri_speech_clean_100hr
#PROFILE=libri_light_10hr
#PROFILE=libri_light_1hr

python wav2vec2_finetuning_sets.py \
    --alpha=0.0  \
    --beta=0.0   \
    --vocab_size 32 \
    --finetune_profile=$PROFILE  \
    --per_device_train_batch_size 24
