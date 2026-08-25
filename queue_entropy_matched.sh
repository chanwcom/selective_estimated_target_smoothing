#!/bin/bash
# Usage: WAIT_PID=<pid> PROFILE=<profile> LOG_DIR=<dir> bash queue_entropy_matched.sh
#
# Waits for the pid holding GPU0 to exit, then runs the C-SETS-H sweep
# (class-space SETS with the smoothing weight solved for by entropy
# matching, rather than tuned) for one fine-tuning profile.
#
# WHY THE SWEEP VALUES LIVE IN A FILE. This script sits in a sleep loop
# for hours before it runs, and bash reads a script incrementally, so
# editing it in place while it waits can corrupt the still-unread
# remainder. The alpha_max values are therefore read from
# `conf_entropy_matched_alpha_max.txt` at the moment the sweep actually
# starts. That file CAN be edited safely at any point beforehand, which
# is the point: the observe-only pilot measures what alpha the entropy
# matching actually selects, and if it comes back far from what we
# assumed, the queued sweep should be retargeted without having to kill
# and relaunch the chain.
#
# The first positional of run_train_dynamic_grid_entropy_matched.sh is
# --entropy_match_alpha_max, not --alpha; run_train_grid_seed_peak_
# capping.py drives it because its sweep axis is likewise a single
# scalar. Its summary.json keys read "alpha=<value>", meaning alpha_max.
set -u

WAIT_PID=${WAIT_PID:-0}
GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_1hr}
LOG_DIR=${LOG_DIR:-grid_logs_1hr_entropy_matched}
CONF=${CONF:-conf_entropy_matched_alpha_max.txt}

cd /mnt/data/home/chanwcom/local_repository/selective_estimated_target_smoothing

if [ "$WAIT_PID" != "0" ]; then
    echo "[queue-hmatch $PROFILE] waiting for pid $WAIT_PID to release GPU$GPU ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[queue-hmatch $PROFILE] pid $WAIT_PID exited at $(date -Is)."
    sleep 30
fi

ALPHA_MAX_VALUES=$(grep -vE '^\s*(#|$)' "$CONF" | tr '\n' ' ')
echo "[queue-hmatch $PROFILE] alpha_max values from $CONF: $ALPHA_MAX_VALUES"

# Pin the interpreter. These queues are launched detached, from a shell
# that has NOT activated the conda environment, so a bare `python`
# resolves to the base install and every run dies instantly on
# `ModuleNotFoundError: No module named 'evaluate'`. Putting the env
# first on PATH keeps the sweep scripts themselves unchanged, so they
# still work when run by hand from an activated shell.
export PATH="/home/chanwcom/miniconda3/envs/py3_10_hf/bin:$PATH"

export CUDA_VISIBLE_DEVICES=$GPU
exec python run_train_grid_seed_peak_capping.py \
    run_train_dynamic_grid_entropy_matched.sh \
    --alphas $ALPHA_MAX_VALUES \
    --seeds 0 1 2 \
    --profile "$PROFILE" \
    --log-dir "$LOG_DIR"
