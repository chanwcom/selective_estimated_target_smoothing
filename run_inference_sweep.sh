#!/bin/bash
# Usage: WORKER=<0..N-1> NWORKERS=<N> GPU=<n> bash run_inference_ls_sweep.sh
#
# Final-performance inference over every classical-LS checkpoint (the
# beta=0 class-space runs, i.e. textbook uniform label smoothing) for both
# libri_light profiles, both LibriSpeech test splits, and both decoders.
#
# Why both decoders: training-time eval_wer is greedy, test-clean only, so
# it is a proxy. The beam-search number is the one to report. Running
# greedy here too -- on the SAME checkpoints and the SAME dataset code
# path as the beam run -- is what makes the greedy/beam gap attributable
# to the decoder rather than to any difference in evaluation setup.
#
# beam_size=40 with log_add=True (see build_beam_search_decoder). log_add
# is what makes beam search differ from greedy at all: with the torchaudio
# default log_add=False the decoder merges hypotheses by max, which
# returns the best PATH, and for a converged CTC model that is the greedy
# answer -- measured 0 of 64 sentences different at every beam size up to
# 500. With log_add=True it sums alignment probabilities, and 7 of 64
# sentences change.
#
# Workers pull from a shared flock-guarded counter rather than taking
# every Nth task. Striding looked simpler but distributes this particular
# list terribly: there are exactly 4 tasks per checkpoint (2 decoders x 2
# splits), so a stride of 4 hands every beam task to two workers and every
# greedy task to the other two. Beam is ~4x slower per task, so those two
# workers would finish in a quarter of the time and then idle -- measured
# 1.2 h vs 4.3 h. With a shared queue every worker takes whatever is next.
#
# Two workers per GPU fit in 24 GB (~5 GB each, measured) and roughly
# double throughput, since a single stream leaves the GPU idle during the
# CPU-side beam decode.
#
# Resumable: a task whose log already holds a result dict is skipped, so
# re-running after an interruption picks up where it left off.
set -u

WORKER=${WORKER:-0}
GPU=${GPU:-0}
QUEUE=${QUEUE:-/tmp/claude-1003/infer_ls_queue}
LOG_DIR=${LOG_DIR:-inference_logs_ls}
BEAM=${BEAM:-40}
BATCH=${BATCH:-8}

cd "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${PYTHONPATH:-}"
source ./set_config.sh
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"

# Build the task list: profile x alpha x seed x decoder x split.
# Checkpoint dirs follow wav2vec2_finetuning_sets.py's _default_run_name().
# Exclude the FAS/ASAP runs -- this sweep is classical LS only.
tasks=()
for PROFILE in libri_light_1hr libri_light_10hr; do
    for RUN in "$CHECKPOINT_TOP_DIR/${PROFILE}_shc_"*"_beta_0p0_unigram_32_dynbatch6400000_seed"?"_classspace"; do
        [ -d "$RUN" ] || continue
        case "$RUN" in *fas*|*asap*) continue;; esac
        # Highest-numbered checkpoint-N, so --max_steps need not be known.
        CK=$(ls -d "$RUN"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        [ -d "$CK" ] || continue
        NAME=$(basename "$RUN")
        ALPHA=$(sed -E 's/.*_alpha_([0-9p]+)_beta.*/\1/' <<<"$NAME")
        SEED=$(sed -E 's/.*_seed([0-9]+)_.*/\1/' <<<"$NAME")
        for DEC in beam_search pipeline; do
            for SPLIT in test-clean test-other; do
                tasks+=("$PROFILE|$ALPHA|$SEED|$DEC|$SPLIT|$CK")
            done
        done
    done
done

echo "[infer w$WORKER/gpu$GPU] $(date -Is) total tasks ${#tasks[@]}, pulling from $QUEUE"

# Atomically take the next index. The lock is held only for the read and
# the write-back, never across a training run, so workers never block on
# each other for more than a few milliseconds.
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

done_n=0; skip_n=0; fail_n=0
while :; do
    i=$(next_index)
    [ "$i" -lt "${#tasks[@]}" ] || break
    IFS='|' read -r PROFILE ALPHA SEED DEC SPLIT CK <<<"${tasks[$i]}"
    LOG="$LOG_DIR/${PROFILE}_alpha${ALPHA}_seed${SEED}_${DEC}_${SPLIT}.log"

    if [ -f "$LOG" ] && grep -q "'wer':" "$LOG"; then
        skip_n=$((skip_n+1))
        continue
    fi

    # beam_size is passed either way; wav2vec2_inference.py ignores it for
    # --decoder=pipeline and records None for it in the result dict.
    if python wav2vec2_inference.py \
            --checkpoint_dir "$CK" \
            --vocab_size 32 \
            --decoder "$DEC" \
            --beam_size "$BEAM" \
            --test_split "$SPLIT" \
            --batch_size "$BATCH" \
            --device cuda > "$LOG" 2>&1; then
        done_n=$((done_n+1))
        echo "[infer w$WORKER] $(date +%H:%M:%S) ok   $PROFILE a=$ALPHA s$SEED $DEC $SPLIT  $(grep -o "'wer': [0-9.]*" "$LOG" | tail -1)"
    else
        fail_n=$((fail_n+1))
        echo "[infer w$WORKER] $(date +%H:%M:%S) FAIL $PROFILE a=$ALPHA s$SEED $DEC $SPLIT -- see $LOG"
    fi
done

echo "[infer w$WORKER/gpu$GPU] $(date -Is) finished: ran $done_n, skipped $skip_n, failed $fail_n"
