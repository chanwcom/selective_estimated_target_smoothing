#!/bin/bash
# Usage: GPU=<n> [WAIT_PID=<pid>] [SEEDS="0 1 2"] bash queue_trunk_branch_1500.sh
#
# Shared-trunk ablation of WHERE label smoothing acts, run on one GPU.
#
#   trunk  (3 runs): alpha=0.1 for steps 0..1500, checkpoint saved at 1500
#   branch (6 runs): resume that checkpoint and run 1500..2000 twice per
#                    seed -- once with smoothing still on (alpha=0.1),
#                    once with it off (alpha=0.0)
#
# Why this is stronger than the O1/O2 schedule arms already run: there, the
# "smooth early" and "smooth late" runs diverged from step 0, so the final
# WER gap mixed the effect of smoothing with ordinary seed-to-seed
# trajectory noise (O1's std was 0.0093, as wide as the gap being measured).
# Here both branches start from the SAME weights, optimizer and scheduler
# state, so the only difference is the smoothing applied over the last 500
# steps -- and the paired difference per seed is directly interpretable.
#
# The trunk keeps --max_steps 2000 and stops via --stop_at_step rather than
# training a 1500-step schedule, so the LR at the branch point is what a
# full 2000-step run would have had. See StopAtStepCallback.
set -u

GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_1hr}
SEEDS=${SEEDS:-"0 1 2"}
ALPHA=${ALPHA:-0.1}
BRANCH_STEP=${BRANCH_STEP:-1500}
WAIT_PID=${WAIT_PID:-0}
LOG_DIR=${LOG_DIR:-grid_logs_1hr_ls_trunk1500}

MAX_BATCH_AUDIO_LEN=${MAX_BATCH_AUDIO_LEN:-6400000}
MAX_SAMPLE_AUDIO_LEN=${MAX_SAMPLE_AUDIO_LEN:-480000}

cd "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$WAIT_PID" != "0" ]; then
    echo "[trunk-branch] waiting for pid $WAIT_PID to release GPU$GPU ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[trunk-branch] pid $WAIT_PID exited at $(date -Is)."
    sleep 30
fi

# CWK's setup_path.sh, reached via set_config.sh, expands $PYTHONPATH
# unguarded, which aborts under `set -u` when this queue is launched
# detached (nohup) from a shell that never exported one.
export PYTHONPATH="${PYTHONPATH:-}"
source ./set_config.sh
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"
export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p "$LOG_DIR"

train() {  # train <run_name> <alpha> <log_name> [extra flags...]
    local run_name=$1 alpha=$2 log_name=$3
    shift 3
    local log="$LOG_DIR/${log_name}.log"
    if grep -q "train_runtime" "$log" 2>/dev/null; then
        echo "[trunk-branch] SKIP $log_name (already complete)"
        return 0
    fi
    echo "[trunk-branch] === $log_name (alpha=$alpha) ==="
    PYTHONUNBUFFERED=1 python wav2vec2_finetuning_sets.py \
        --alpha="$alpha" \
        --beta=0.0 \
        --smoothing_space=class \
        --vocab_size 32 \
        --finetune_profile="$PROFILE" \
        --run_name="$run_name" \
        --dynamic_batching \
        --max_batch_audio_len $MAX_BATCH_AUDIO_LEN \
        --max_sample_audio_len $MAX_SAMPLE_AUDIO_LEN \
        --dataloader_num_workers 4 \
        --dataloader_persistent_workers \
        "$@" 2>&1 | tee "$log"
}

for seed in $SEEDS; do
    trunk_name="ls_trunk_a$(echo $ALPHA | tr . p)_stop${BRANCH_STEP}_seed${seed}"
    ckpt="$CHECKPOINT_TOP_DIR/$trunk_name/checkpoint-$BRANCH_STEP"

    train "$trunk_name" "$ALPHA" "trunk_seed${seed}" \
        --seed "$seed" \
        --stop_at_step "$BRANCH_STEP" \
        --save_steps "$BRANCH_STEP"

    if [ ! -d "$ckpt" ]; then
        echo "[trunk-branch] ERROR: $ckpt missing; skipping seed $seed branches" >&2
        continue
    fi

    train "ls_branch_on_a$(echo $ALPHA | tr . p)_res${BRANCH_STEP}_seed${seed}" \
        "$ALPHA" "branch_on_seed${seed}" \
        --seed "$seed" --resume_from_checkpoint "$ckpt"

    train "ls_branch_off_a0p0_res${BRANCH_STEP}_seed${seed}" \
        0.0 "branch_off_seed${seed}" \
        --seed "$seed" --resume_from_checkpoint "$ckpt"
done

echo "[trunk-branch] all done at $(date -Is)."
