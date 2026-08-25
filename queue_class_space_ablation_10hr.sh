#!/bin/bash
# 10hr counterpart of queue_class_space_ablation.sh. Waits for the 1hr
# class-space ablation to release GPU0, then runs the same label-space vs
# class-space comparison on the 10hr fine-tuning set.
#
# The config differs from the 1hr run on purpose. Each data size is compared
# at ITS OWN best label-space setting, so the question stays "at the
# operating point that already works here, does the smoothing space
# matter?" rather than "does one fixed alpha survive a change of space?".
# From the completed label-space grids (greedy proxy eval_wer, n=3):
#
#     1hr  best: alpha=0.02, beta=1.0 -> 0.2003 +/- 0.0025
#     10hr best: alpha=0.01, beta=1.0 -> 0.0992 +/- 0.0007
#
# (10hr's alpha=0.02, beta=1.0 is 0.1004 +/- 0.0020, outside the top 4.)
# Note that the 10hr grid is fairly flat at the top -- its top 4 cells span
# 0.0992..0.1000, comfortably inside the seed-to-seed spread -- so read a
# small 10hr difference with that in mind.
#
# A separate file rather than an argument to queue_class_space_ablation.sh
# because that script is mid-execution while this one is written, and bash
# reads a running script incrementally: editing it in place can corrupt the
# still-unread remainder.
set -u

WAIT_PID=${WAIT_PID:-0}
GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_10hr}
LOG_DIR=${LOG_DIR:-grid_logs_10hr_class_space}
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
    echo "[queue-10hr] waiting for pid $WAIT_PID to release GPU$GPU ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[queue-10hr] pid $WAIT_PID exited at $(date -Is); starting ablation."
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
