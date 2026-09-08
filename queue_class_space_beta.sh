#!/bin/bash
# Usage: ALPHAS="..." BETAS="..." WAIT_PID=<pid> GPU=<n> PROFILE=<p> \
#            LOG_DIR=<dir> bash queue_class_space_beta.sh
#
# C-SETS beta sweep: --smoothing_space=class with an arbitrary BETAS list,
# unlike queue_uniform_ls.sh (same underlying
# run_train_dynamic_grid_class_space.sh) which hardcodes --betas 0.0 and
# was built only for the classical-LS beta=0 case. That hardcoding is not
# a bug there -- it is what makes it the classical-LS queue -- but it
# means queue_uniform_ls.sh silently ignores any BETAS env var passed to
# it, which is exactly the mistake that produced this file: three
# separate launches of a "C-SETS beta sweep" via queue_uniform_ls.sh with
# BETAS="0.25 0.5 0.75 ..." actually queued more beta=0.0 (classical LS)
# reruns, caught only after one of them started training on GPU1.
set -u

WAIT_PID=${WAIT_PID:-0}
GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_1hr}
LOG_DIR=${LOG_DIR:-grid_logs_1hr_class_space}
ALPHAS=${ALPHAS:-"0.01 0.02 0.03 0.04 0.05"}
BETAS=${BETAS:-"0.25 0.5 0.75"}

cd "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$WAIT_PID" != "0" ]; then
    echo "[queue-csets $PROFILE] waiting for pid $WAIT_PID to release GPU$GPU ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[queue-csets $PROFILE] pid $WAIT_PID exited at $(date -Is)."
    sleep 30
fi

# Pin the interpreter -- these queues launch detached from a shell with no
# conda environment activated, where a bare `python` is the base install
# and every run dies on `ModuleNotFoundError: No module named 'evaluate'`.
source ./set_config.sh
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"

export CUDA_VISIBLE_DEVICES=$GPU
echo "[queue-csets $PROFILE] alphas: $ALPHAS"
echo "[queue-csets $PROFILE] betas:  $BETAS"
exec python run_train_grid_seed.py \
    run_train_dynamic_grid_class_space.sh \
    --alphas $ALPHAS \
    --betas $BETAS \
    --seeds 0 1 2 \
    --profile "$PROFILE" \
    --log-dir "$LOG_DIR"
