#!/bin/bash
# C-SETS-SH queue: entropy matching with the reference entropy restricted to
# the active set, no alpha_max. See run_train_dynamic_grid_sh.sh for why the
# cap is deliberately absent.
set -u
WAIT_PID=${WAIT_PID:-0}
GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_1hr}
LOG_DIR=${LOG_DIR:-grid_logs_1hr_sh}
cd /mnt/data/home/chanwcom/local_repository/selective_estimated_target_smoothing
if [ "$WAIT_PID" != "0" ]; then
    echo "[queue-sh $PROFILE] waiting for pid $WAIT_PID ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[queue-sh $PROFILE] pid $WAIT_PID exited at $(date -Is)."
    sleep 30
fi
export PATH="/home/chanwcom/miniconda3/envs/py3_10_hf/bin:$PATH"
export CUDA_VISIBLE_DEVICES=$GPU
exec python run_train_grid_seed_peak_capping.py \
    run_train_dynamic_grid_sh.sh --alphas 0 --seeds 0 1 2 \
    --profile "$PROFILE" --log-dir "$LOG_DIR"
