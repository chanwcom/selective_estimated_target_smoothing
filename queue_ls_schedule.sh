#!/bin/bash
# Usage: GPU=<n> ALPHA=<a> ALPHA_AFTER=<a'> [ALPHA_SWITCH_STEP=1000] \
#            [LOG_DIR=<dir>] bash queue_ls_schedule.sh
#
# One arm of the label-smoothing timing experiment: classical uniform label
# smoothing (class space, beta=0 -- see queue_uniform_ls.sh) held at ALPHA
# until ALPHA_SWITCH_STEP, then at ALPHA_AFTER for the rest of the run.
#
# The pair worth running is the two orderings of the same two values, so
# total smoothing exposure is comparable and only its POSITION in training
# differs:
#   early-only: ALPHA=0.1 ALPHA_AFTER=0.0   (regularize, then anneal away)
#   late-only:  ALPHA=0.0 ALPHA_AFTER=0.1   (sharpen first, then calibrate)
# At libri_light_1hr's 2000-step schedule, a switch at 1000 splits training
# exactly in half.
set -u

GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_1hr}
ALPHA=${ALPHA:?set ALPHA (smoothing weight before the switch)}
ALPHA_AFTER=${ALPHA_AFTER:?set ALPHA_AFTER (smoothing weight after it)}
ALPHA_SWITCH_STEP=${ALPHA_SWITCH_STEP:-1000}
SEEDS=${SEEDS:-"0 1 2"}
LOG_DIR=${LOG_DIR:-grid_logs_1hr_ls_schedule}

cd "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CWK's setup_path.sh, reached via set_config.sh, expands $PYTHONPATH
# unguarded, which aborts under `set -u` when this queue is launched
# detached (nohup) from a shell that never exported one.
export PYTHONPATH="${PYTHONPATH:-}"

# Pin the interpreter -- these queues launch detached from a shell with no
# conda environment activated, where a bare `python` is the base install.
source ./set_config.sh
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"

export CUDA_VISIBLE_DEVICES=$GPU
export ALPHA_AFTER ALPHA_SWITCH_STEP

echo "[queue-ls-sched $PROFILE] gpu=$GPU alpha=$ALPHA -> $ALPHA_AFTER at step $ALPHA_SWITCH_STEP, seeds: $SEEDS"
exec python run_train_grid_seed.py \
    run_train_dynamic_grid_ls_schedule.sh \
    --alphas $ALPHA \
    --betas 0.0 \
    --seeds $SEEDS \
    --profile "$PROFILE" \
    --log-dir "$LOG_DIR"
