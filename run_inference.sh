python wav2vec2_inference.py \
    --checkpoint_dir /mnt/data/home/chanwcom/models/shc_2000steps_alpha_0p0_beta_0p0_unigram_32/checkpoint-2000 \
    --vocab_size 32 \
    --decoder beam_search \
    --beam_size 80
