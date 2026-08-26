#!/bin/bash
# Usage: ALPHAS="0.01 0.02" WAIT_PID=<pid> GPU=<n> PROFILE=<p> bash queue_uniform_ls.sh
#
# Classical uniform label smoothing: (1-alpha)*y + alpha/C, mass on every
# one of the C classes including those absent from the transcript.
#
# This is the baseline every reviewer asks for and it has never been run.
# The grids so far used label space, where beta=0's "uniform" is 1/L over
# blank-augmented label POSITIONS, which lands in class space as ~0.49 on
# blank, occurrence-weighted mass on only the classes present in the
# transcript, and exactly 0 on the rest -- unigram smoothing with a blank
# prior, not uniform smoothing. Verified numerically: class space with
# beta=0 reproduces (1-alpha)*y + alpha/C to 0.00e+00, and puts alpha/C on
# absent classes; label space with beta=0 does not.
#
# No new training code is needed -- run_train_dynamic_grid_class_space.sh
# already takes beta as its second positional, so beta=0.0 there is exactly
# textbook label smoothing.
set -u

WAIT_PID=${WAIT_PID:-0}
GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_1hr}
LOG_DIR=${LOG_DIR:-grid_logs_1hr_uniform_ls}
ALPHAS=${ALPHAS:-"0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10"}

cd /mnt/data/home/chanwcom/local_repository/selective_estimated_target_smoothing

if [ "$WAIT_PID" != "0" ]; then
    echo "[queue-ls $PROFILE] waiting for pid $WAIT_PID to release GPU$GPU ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[queue-ls $PROFILE] pid $WAIT_PID exited at $(date -Is)."
    sleep 30
fi

# Pin the interpreter -- these queues launch detached from a shell with no
# conda environment activated, where a bare `python` is the base install.
export PATH="/home/chanwcom/miniconda3/envs/py3_10_hf/bin:$PATH"

export CUDA_VISIBLE_DEVICES=$GPU
echo "[queue-ls $PROFILE] alphas: $ALPHAS  (beta=0.0, class space)"
exec python run_train_grid_seed.py \
    run_train_dynamic_grid_class_space.sh \
    --alphas $ALPHAS \
    --betas 0.0 \
    --seeds 0 1 2 \
    --profile "$PROFILE" \
    --log-dir "$LOG_DIR"
