#!/bin/bash
# Usage: GPU=<n> PROFILE=<p> BRANCH_STEP=<n> [WAIT_PID=<pid>] [SEEDS="0 1 2"]
#            [ALPHA=0.1] [LOG_DIR=<dir>] bash queue_trunk_branch.sh
#
# Shared-trunk ablation of WHERE label smoothing acts, run on one GPU.
#
#   trunk  (1 run per seed): alpha=ALPHA for steps 0..BRANCH_STEP, with a
#                            checkpoint saved at BRANCH_STEP
#   branch (2 runs per seed): resume that checkpoint and finish the
#                            profile's schedule twice -- once with
#                            smoothing still on (ALPHA), once off (0.0)
#
# Why this beats comparing two independent runs: schedule arms that diverge
# at step 0 mix the effect of smoothing with ordinary seed-to-seed
# trajectory noise. On libri_light_1hr the "smooth early" arm's seed spread
# (std 0.0093) was as wide as the effect being measured. Here both branches
# start from the SAME weights, optimizer and scheduler state, so the paired
# per-seed difference isolates the smoothing applied over the tail.
#
# The trunk keeps the profile's full --max_steps and stops via
# --stop_at_step rather than training a shortened schedule, so the LR at
# the branch point is what a full-length run would have had. See
# StopAtStepCallback in wav2vec2_finetuning_sets.py.
#
# Sizing per profile: libri_light_1hr is 2000 steps (branch at 1500),
# libri_light_10hr is 4000 (branch at 3000). Both use the 4090 gpu_profile
# and the same dynamic-batching budget as run_train_dynamic_grid*.sh.
set -u

GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_1hr}
SEEDS=${SEEDS:-"0 1 2"}
ALPHA=${ALPHA:-0.1}
BRANCH_STEP=${BRANCH_STEP:?set BRANCH_STEP (step to save the trunk checkpoint at)}
WAIT_PID=${WAIT_PID:-0}
LOG_DIR=${LOG_DIR:-grid_logs_${PROFILE}_ls_trunk${BRANCH_STEP}}

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
A=$(echo "$ALPHA" | tr . p)

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
    # PROFILE is part of every run name: without it a 10hr branch-at-3000
    # and a 1hr branch-at-3000 would share a checkpoint directory.
    trunk_name="${PROFILE}_ls_trunk_a${A}_stop${BRANCH_STEP}_seed${seed}"
    ckpt="$CHECKPOINT_TOP_DIR/$trunk_name/checkpoint-$BRANCH_STEP"

    train "$trunk_name" "$ALPHA" "trunk_seed${seed}" \
        --seed "$seed" \
        --stop_at_step "$BRANCH_STEP" \
        --save_steps "$BRANCH_STEP"

    if [ ! -d "$ckpt" ]; then
        echo "[trunk-branch] ERROR: $ckpt missing; skipping seed $seed branches" >&2
        continue
    fi

    train "${PROFILE}_ls_branch_on_a${A}_res${BRANCH_STEP}_seed${seed}" \
        "$ALPHA" "branch_on_seed${seed}" \
        --seed "$seed" --resume_from_checkpoint "$ckpt"

    train "${PROFILE}_ls_branch_off_a0p0_res${BRANCH_STEP}_seed${seed}" \
        0.0 "branch_off_seed${seed}" \
        --seed "$seed" --resume_from_checkpoint "$ckpt"
done

echo "[trunk-branch] all done at $(date -Is)."
