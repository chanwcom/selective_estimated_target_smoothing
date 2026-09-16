#!/bin/bash
# Usage: PATTERN='<glob>' LOG_DIR=<dir> WORKER=<n> bash run_inference_sweep.sh
#
# Final-performance inference over every checkpoint matching PATTERN, for
# both LibriSpeech test splits and both decoders.
#
# Why both decoders: training-time eval_wer is greedy, test-clean only, so
# it is a proxy. The beam-search number is the one to report. Running
# greedy here too -- on the SAME checkpoints and the SAME dataset code path
# as the beam run -- is what makes the greedy/beam gap attributable to the
# decoder rather than to any difference in evaluation setup.
#
# beam_size=40 with log_add=True (set in build_beam_search_decoder). log_add
# is what makes beam search differ from greedy at all: with torchaudio's
# default log_add=False the decoder merges hypotheses by max, which returns
# the best PATH, and for a converged CTC model that is the greedy answer --
# measured 0 of 64 sentences different at every beam size up to 500. With
# log_add=True it sums alignment probabilities and 7 of 64 sentences change.
#
# DEVICE defaults to cpu, and that is the right default while the GPUs are
# training. The beam decode is CPU-only anyway (torchaudio's ctc_decoder has
# no CUDA path -- decode_batch_beam_search moves log_probs to the host), so
# the GPU only accelerates the wav2vec2 forward, which runs at ~29x realtime
# on these cores. Each worker is single-threaded in the decoder, so parallel
# workers, not threads, are what buys throughput.
#
# THREADS caps torch's intra-op pool per worker, and CPU_LIST hard-pins the
# worker to a taskset. Both are needed to stay out of a concurrent training
# run's way, and `nice` alone is not enough: with 6 workers x 4 threads
# unpinned on a 40-core box the training jobs went from 1.50 to 2.15 s/it,
# a 43% slowdown, while their own CPU draw never moved (236%) and GPU
# utilisation fell to 5% -- they were waiting, not computing. OpenMP threads
# spin-wait by default, so they keep a core busy without being runnable work
# that nice can deprioritise, and the training dataloaders lose the race for
# a core. OMP_WAIT_POLICY=PASSIVE below stops the spinning; CPU_LIST makes
# the isolation absolute by leaving a block of cores the workers cannot
# touch at all.
#
# Work comes from a shared flock-guarded counter rather than a stride.
# Striding distributes this list terribly: there are exactly 4 tasks per
# checkpoint (2 decoders x 2 splits) and beam is ~4x slower than greedy, so
# a stride of 4 hands every beam task to one worker and every greedy task to
# another -- measured 1.2 h vs 4.3 h for the same share.
#
# Resumable: a task whose log already holds a result dict is skipped.
#
# NOTE: inference_logs_ls/ was produced by this script before it took
# PATTERN, when the log name had no beta field (every LS run has beta=0).
# Those names are one field shorter than what this version writes; point
# LOG_DIR somewhere else rather than mixing the two conventions.
set -u

WORKER=${WORKER:-0}
QUEUE=${QUEUE:-/tmp/claude-1003/infer_queue}
PATTERN=${PATTERN:?"set PATTERN, e.g. '*_10hr_*fas_eps*'"}
LOG_DIR=${LOG_DIR:-inference_logs}
BEAM=${BEAM:-40}
BATCH=${BATCH:-8}
DEVICE=${DEVICE:-cpu}
THREADS=${THREADS:-4}
NICE=${NICE:-19}
# Cores this worker may use, as a taskset list (e.g. "28-39"). Empty = no
# pinning. See the THREADS note above for why pinning is not optional when
# a training run shares the box.
CPU_LIST=${CPU_LIST:-}

cd "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${PYTHONPATH:-}"
source ./set_config.sh
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
# Stop OpenMP from spin-waiting between parallel regions; a spinning thread
# holds a core without being work `nice` can push aside.
export OMP_WAIT_POLICY=PASSIVE
# set_config.sh pins CUDA_VISIBLE_DEVICES to DEVICE_ID; blank it for CPU
# runs so a worker cannot take memory on a card that is training.
[ "$DEVICE" = "cpu" ] && export CUDA_VISIBLE_DEVICES=""

PIN=""
[ -n "$CPU_LIST" ] && PIN="taskset -c $CPU_LIST"

mkdir -p "$LOG_DIR"

tasks=()
for RUN in $CHECKPOINT_TOP_DIR/$PATTERN; do
    [ -d "$RUN" ] || continue
    # Highest-numbered checkpoint-N, so --max_steps need not be known here.
    CK=$(ls -d "$RUN"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [ -d "$CK" ] || continue
    NAME=$(basename "$RUN")
    PROF=$(sed -E 's/.*(libri_light_[0-9]+hr|libri_speech[a-z_0-9]*).*/\1/' <<<"$NAME")
    ALPHA=$(sed -E 's/.*_alpha_([0-9p]+)_.*/\1/' <<<"$NAME")
    BETA=$(sed -E 's/.*_beta_([0-9p]+)_.*/\1/' <<<"$NAME")
    SEED=$(sed -E 's/.*_seed([0-9]+).*/\1/' <<<"$NAME")
    for DEC in beam_search pipeline; do
        for SPLIT in test-clean test-other; do
            tasks+=("$PROF|$ALPHA|$BETA|$SEED|$DEC|$SPLIT|$CK")
        done
    done
done

echo "[infer w$WORKER/$DEVICE] $(date -Is) ${#tasks[@]} tasks from '$PATTERN', queue $QUEUE"

next_index() {
    local i
    exec 9>"$QUEUE.lock"
    flock 9
    i=$(cat "$QUEUE" 2>/dev/null || echo 0)
    echo $((i + 1)) > "$QUEUE"
    flock -u 9
    exec 9>&-
    echo "$i"
}

ran=0; skipped=0; failed=0
while :; do
    i=$(next_index)
    [ "$i" -lt "${#tasks[@]}" ] || break
    IFS='|' read -r PROF ALPHA BETA SEED DEC SPLIT CK <<<"${tasks[$i]}"
    LOG="$LOG_DIR/${PROF}_alpha${ALPHA}_beta${BETA}_seed${SEED}_${DEC}_${SPLIT}.log"

    if [ -f "$LOG" ] && grep -q "'wer':" "$LOG"; then
        skipped=$((skipped + 1))
        continue
    fi

    # beam_size is passed either way; wav2vec2_inference.py ignores it for
    # --decoder=pipeline and records None for it in the result dict.
    if $PIN nice -n "$NICE" python wav2vec2_inference.py \
            --checkpoint_dir "$CK" \
            --vocab_size 32 \
            --decoder "$DEC" \
            --beam_size "$BEAM" \
            --test_split "$SPLIT" \
            --batch_size "$BATCH" \
            --device "$DEVICE" > "$LOG" 2>&1; then
        ran=$((ran + 1))
        echo "[infer w$WORKER] $(date +%H:%M:%S) ok   a=$ALPHA b=$BETA s$SEED $DEC $SPLIT  $(grep -o "'wer': [0-9.]*" "$LOG" | tail -1)"
    else
        failed=$((failed + 1))
        echo "[infer w$WORKER] $(date +%H:%M:%S) FAIL a=$ALPHA b=$BETA s$SEED $DEC $SPLIT -- see $LOG"
    fi
done

echo "[infer w$WORKER/$DEVICE] $(date -Is) done: ran $ran, skipped $skipped, failed $failed"
