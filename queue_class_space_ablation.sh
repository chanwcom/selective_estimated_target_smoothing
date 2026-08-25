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
# Swept, not fixed. 0.02 is the LABEL-space optimum, and the two spaces
# do not share an alpha scale: the intervention magnitude is
# alpha * ||m - q~||, and those norms differ (measured mean 0.997 in
# label space vs 1.145 in class space over 26k frames, ratio 0.871, so
# label alpha=0.02 is worth about 0.0174 here). Running class space at
# label space's optimum would confound "class space is worse" with
# "0.02 is not its optimum", so each space is compared at its own best
# cell.
ALPHAS=${ALPHAS:-"0.01 0.02 0.03"}
BETA=${BETA:-1.0}

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

# Pin the interpreter. These queues are launched detached, from a shell
# that has NOT activated the conda environment, so a bare `python`
# resolves to the base install and every run dies instantly on
# `ModuleNotFoundError: No module named 'evaluate'`. Putting the env
# first on PATH keeps the sweep scripts themselves unchanged, so they
# still work when run by hand from an activated shell.
export PATH="/home/chanwcom/miniconda3/envs/py3_10_hf/bin:$PATH"

export CUDA_VISIBLE_DEVICES=$GPU
exec python run_train_grid_seed.py \
    run_train_dynamic_grid_class_space.sh \
    --alphas $ALPHAS \
    --betas "$BETA" \
    --seeds 0 1 2 \
    --profile "$PROFILE" \
    --log-dir "$LOG_DIR"
