#!/bin/bash
# Waits for the GPU0 sweep currently in flight to finish, then runs the
# class-space SETS ablation on GPU0.
#
# The ablation is deliberately narrow: alpha=0.02, beta=1.0 only -- the
# configuration that won the label-space 1hr grid -- repeated over 3 seeds.
# The question it answers is not "what is the best class-space setting" but
# "at the setting that already works, does smoothing over classes beat
# smoothing over label positions?". See run_train_dynamic_grid_class_space.sh
# for why that question matters.
#
# WAIT_PID defaults to the SETS alpha=0.05 sweep that owns GPU0 as of
# launch. Override it (or set it to 0 to skip waiting) if the situation has
# changed since.
set -u

WAIT_PID=${WAIT_PID:-203453}
GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_1hr}
LOG_DIR=${LOG_DIR:-grid_logs_1hr_class_space}

cd /mnt/data/home/chanwcom/local_repository/selective_estimated_target_smoothing

if [ "$WAIT_PID" != "0" ]; then
    echo "[queue] waiting for pid $WAIT_PID to release GPU$GPU ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[queue] pid $WAIT_PID exited at $(date -Is); starting ablation."
    # Let the GPU memory actually drain before claiming it.
    sleep 30
fi

export CUDA_VISIBLE_DEVICES=$GPU
exec python run_train_grid_seed.py \
    run_train_dynamic_grid_class_space.sh \
    --alphas 0.02 \
    --betas 1.0 \
    --seeds 0 1 2 \
    --profile "$PROFILE" \
    --log-dir "$LOG_DIR"
