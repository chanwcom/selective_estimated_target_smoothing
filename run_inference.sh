# Usage: bash run_inference.sh <checkpoint_dir>
#
# Runs final WER evaluation on BOTH LibriSpeech test splits (test-clean,
# test-other) -- training-time eval only ever covered test-clean, so this
# is the actual final-performance number, not the periodic training eval.
#
# beam_size=20: matches what training-side inference already settled on
# (see run_train_fixed.sh's inference note / earlier "reducing the
# default beam size" commit) -- CTC beam search without an LM has fast-
# diminishing returns past ~20, so this trades a bit of possible WER gain
# for meaningfully faster evaluation. Bump it back up if you specifically
# want to check whether a larger beam changes the ranking between configs.
CHECKPOINT_DIR=${1:?"Usage: bash run_inference.sh <checkpoint_dir>"}

for SPLIT in test-clean test-other; do
    python wav2vec2_inference.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --vocab_size 32 \
        --decoder beam_search \
        --beam_size 20 \
        --test_split $SPLIT
done
