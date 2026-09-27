#!/bin/bash
# run_pas.sh <baseline|pas_h|pas_u> <ctc|rnnt> <alpha> <seed> <gpu>
#
# One run, start to finish: train, evaluate on all four splits with no
# length filter, append one row to RESULT_PAS.md. The three steps live in
# one script because splitting them is how 867 runs ended up with 347 test
# numbers -- evaluation that is "done later" is not done.
#
# The batch budget is fixed at 1600000 and is not a parameter. It was
# measured: RNN-T peaks at 14.42 GiB reserved with eval included, which is
# two per 5090 and one per 4090, and CTC uses the same value so the two
# tables read under the same conditions. See PAS_SPEC.md section 4.2.
set -u
export PYTHONPATH="${PYTHONPATH:-}"

METHOD=$1; LOSS=$2; ALPHA=$3; SEED=$4; GPU=$5

TOP=/mnt/synology_nas_00/chanwcom
MODELS=$TOP/models
LOGS=$TOP/logs
RESULTS=$TOP/results/RESULT_PAS.md
REPO=/mnt/synology_nas_00/chanwcom/local_repository/selective_estimated_target_smoothing

if [ "$METHOD" = "baseline" ]; then
    NAME=baseline_${LOSS}_libri100hr_s${SEED}
    MODE_ARGS="--alpha_mode fixed --alpha 0.0"
    PRETTY=baseline
    ALPHA=0.0
else
    AP=$(echo "$ALPHA" | tr '.' 'p')
    NAME=${METHOD}_${LOSS}_libri100hr_alpha_${AP}_s${SEED}
    MODE_ARGS="--alpha_mode ${METHOD} --alpha ${ALPHA}"
    PRETTY=$(echo "$METHOD" | tr 'a-z_' 'A-Z-')
fi

CKPT=$MODELS/$NAME
LOG=$LOGS/$NAME.log

cd "$REPO"
source /home/chanwcom/miniconda3/etc/profile.d/conda.sh
conda activate py3_12_sets
source set_config.sh
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.7

SCHED="--max_steps 15000 --warmup_steps 1000 --num_stable_steps 11000 --num_decay_steps 3000"
BATCH="--dynamic_batching --max_batch_audio_len 1600000 --max_sample_audio_len 480000"
COMMON="--model_name facebook/wav2vec2-large-lv60 --fas_eps 1e-10 --vocab_size 32 \
 --seed $SEED --finetune_profile libri_speech_clean_100hr_wsd $SCHED \
 --learning_rate 5e-5 $BATCH --dataloader_num_workers 4 --gpu_memory_fraction 0.46"

echo "=== $NAME on GPU $GPU ===" | tee "$LOG"

# ---------------------------------------------------------------- train
if [ "$LOSS" = "ctc" ]; then
    python wav2vec2_finetuning_pas.py $COMMON $MODE_ARGS \
        --smoothing_space label --per_device_eval_batch_size 4 \
        --checkpoint_top_dir "$MODELS" --run_name "$NAME" >> "$LOG" 2>&1
else
    python wav2vec2_rnnt.py $COMMON $MODE_ARGS \
        --max_label_len 450 --eval_batch_size 4 --grad_accum 2 \
        --output_dir "$CKPT" >> "$LOG" 2>&1
fi
RC=$?
if [ $RC -ne 0 ]; then
    echo "TRAIN FAILED rc=$RC -- see $LOG" | tee -a "$LOG"
    exit $RC
fi

# ----------------------------------------------------------------- eval
# No length filter anywhere here. Full dev is 2703/2864 and full test is
# 2620/2939; n is recorded so a filtered number cannot pass as a clean one.
echo "=== eval $NAME ===" >> "$LOG"
DEC=$LOGS/${NAME}_decode.jsonl
: > "$DEC"

if [ "$LOSS" = "ctc" ]; then
    for SPLIT in dev-clean dev-other test-clean test-other; do
        python wav2vec2_inference.py --checkpoint_dir "$CKPT" \
            --vocab_size 32 --test_split "$SPLIT" --batch_size 8 \
            2>&1 | tee -a "$LOG" | python3 -c "
import ast, json, sys
for line in sys.stdin:
    line = line.strip()
    if line.startswith('{') and \"'wer'\" in line:
        d = ast.literal_eval(line)
        print(json.dumps({'split': d['test_split'], 'wer': d['wer'],
                          'n': d['num_examples']}))
" >> "$DEC"
    done
else
    python rnnt_decode_test.py --device cuda --batch_size 8 \
        --splits dev-clean,dev-other,test-clean,test-other \
        --ckpt_glob "$CKPT/rnnt.pt" --out "$DEC" >> "$LOG" 2>&1
fi

# ---------------------------------------------------------------- record
python3 - "$RESULTS" "$DEC" "$PRETTY" "$LOSS" "$ALPHA" "$SEED" "$CKPT" <<'PY'
import json, os, sys
results, dec, method, loss, alpha, seed, ckpt = sys.argv[1:8]

vals = {}
with open(dec) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        vals[d["split"]] = (d["wer"], d["n"])

def cell(split):
    if split not in vals:
        return "—"
    w, n = vals[split]
    return f"{100 * w:.2f} (n={n})"

if not os.path.exists(results):
    with open(results, "w") as f:
        f.write("# PAS sweep results\n\n"
                "Batch inference, no length filter. Model "
                "`facebook/wav2vec2-large-lv60`, LibriSpeech train-clean-100,\n"
                "15,000 steps, WSD 1000/11000/3000, lr 1e-4, "
                "`--max_batch_audio_len 1600000`.\n\n"
                "WER in percent. Full dev is n=2703/2864 and full test is "
                "n=2620/2939 — any other n means a\nfilter was applied and "
                "the row cannot go in a table with the others.\n\n"
                "| method | loss | alpha | seed | finetune set | dev-clean | "
                "dev-other | test-clean | test-other | checkpoint |\n"
                "|---|---|---|---|---|---|---|---|---|---|\n")

with open(results, "a") as f:
    f.write(f"| {method} | {loss.upper()} | {alpha} | {seed} | "
            f"LibriSpeech 100h | {cell('dev-clean')} | {cell('dev-other')} | "
            f"{cell('test-clean')} | {cell('test-other')} | `{ckpt}` |\n")
print(f"recorded {method} {loss} alpha={alpha} seed={seed}")
PY
echo "=== $NAME DONE ===" | tee -a "$LOG"
