#!/bin/bash
# Usage: ALPHAS="..." BETAS="..." WAIT_PID=<pid> GPU=<n> PROFILE=<p> \
#            LOG_DIR=<dir> bash queue_hybrid_space.sh
#
# Runs the alpha x beta grid for the corrected L-SETS, i.e.
# --smoothing_space=hybrid: support chosen on the blank-augmented label
# axis, uniform taken over the class axis. See
# run_train_dynamic_grid_hybrid_space.sh for what was verified before
# this was queued, and cwk/loss/pytorch/shc_loss_hybrid_space_test.py for
# the assertions themselves.
#
# BETA ORDER MATTERS HERE and is deliberately not ascending. beta = 1
# should reproduce the existing L-SETS numbers and beta = 0 the existing
# uniform-LS numbers -- both already measured -- so running beta = 1 FIRST
# turns the first finished cell into an end-to-end check of a freshly
# written code path, available in ~35 min (1hr profile) rather than after
# the whole grid. Reference cells to check it against:
#
#     1hr   beta=1  alpha=0.02  L-SETS      0.2003 +/- 0.0025
#     10hr  beta=1  alpha=0.01  L-SETS      0.0992 +/- 0.0007
#     1hr   beta=0  alpha=0.03  uniform LS  0.2113 +/- 0.0064
#
# The intermediate betas are the genuinely new data: no run has ever put a
# real class uniform at the beta = 0 end, so 0.25 / 0.5 / 0.75 have never
# meant what they mean here. beta = 0 closes the grid as the second check.
set -u

WAIT_PID=${WAIT_PID:-0}
GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_1hr}
LOG_DIR=${LOG_DIR:-grid_logs_1hr_hybrid}
ALPHAS=${ALPHAS:-"0.01 0.02 0.03 0.04 0.05"}
BETAS=${BETAS:-"1.0 0.5 0.25 0.75 0.0"}

cd "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$WAIT_PID" != "0" ]; then
    echo "[queue-hybrid $PROFILE] waiting for pid $WAIT_PID to release GPU$GPU ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[queue-hybrid $PROFILE] pid $WAIT_PID exited at $(date -Is)."
    sleep 30
fi

# Pin the interpreter -- these queues launch detached from a shell with no
# conda environment activated, where a bare `python` is the base install
# and every run dies on `ModuleNotFoundError: No module named 'evaluate'`.
source ./set_config.sh
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"

export CUDA_VISIBLE_DEVICES=$GPU
echo "[queue-hybrid $PROFILE] alphas: $ALPHAS"
echo "[queue-hybrid $PROFILE] betas:  $BETAS"
exec python run_train_grid_seed.py \
    run_train_dynamic_grid_hybrid_space.sh \
    --alphas $ALPHAS \
    --betas $BETAS \
    --seeds 0 1 2 \
    --profile "$PROFILE" \
    --log-dir "$LOG_DIR"
