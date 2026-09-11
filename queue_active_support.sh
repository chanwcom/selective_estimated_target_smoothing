#!/bin/bash
# Usage: GPU=<n> ALPHAS="0.05 0.1" [PROFILE=<p>] [LOG_DIR=<dir>]
#            [WAIT_PID=<pid>] bash queue_active_support.sh
#
# Sweeps alpha for active-support smoothing (AS). See
# run_train_dynamic_grid_active_support.sh for what the method is.
#
# Give each concurrent GPU its own LOG_DIR: run_train_grid_seed.py
# read-modify-writes summary.json in there, so two sweeps sharing one
# directory would clobber each other's cells.
set -u

WAIT_PID=${WAIT_PID:-0}
GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_1hr}
ALPHAS=${ALPHAS:?set ALPHAS (space-separated per-class smoothing rates)}
SEEDS=${SEEDS:-"0 1 2"}
LOG_DIR=${LOG_DIR:-grid_logs_1hr_active_support}

cd "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$WAIT_PID" != "0" ]; then
    echo "[queue-as $PROFILE] waiting for pid $WAIT_PID to release GPU$GPU ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[queue-as $PROFILE] pid $WAIT_PID exited at $(date -Is)."
    sleep 30
fi

# CWK's setup_path.sh, reached via set_config.sh, expands $PYTHONPATH
# unguarded, which aborts under `set -u` when launched detached (nohup)
# from a shell that never exported one.
export PYTHONPATH="${PYTHONPATH:-}"
source ./set_config.sh
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"

export CUDA_VISIBLE_DEVICES=$GPU
echo "[queue-as $PROFILE] gpu=$GPU alphas: $ALPHAS  seeds: $SEEDS"
exec python run_train_grid_seed.py \
    run_train_dynamic_grid_active_support.sh \
    --alphas $ALPHAS \
    --betas 0.0 \
    --seeds $SEEDS \
    --profile "$PROFILE" \
    --log-dir "$LOG_DIR"
