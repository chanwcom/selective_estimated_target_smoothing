#!/bin/bash
# Usage: GPU=<n> ALPHAS="..." BETAS="..." [ASAP_EPS=1e-5]
#            [SEEDS="0 1 2"] [PROFILE=<p>] LOG_DIR=<dir>
#            [WAIT_PID=<pid>] bash queue_fas.sh
#
# Sweeps Floored Active Support. alpha sets the per-class rate on the active
# set, beta controls the floor the inactive classes get (beta*alpha/C).
# beta=0 is plain AS and beta=1 is textbook uniform LS, both bit-exact, so
# one grid spans both endpoints.
#
# ASAP_EPS applies to every cell in one invocation, so a sweep that
# varies it needs one invocation per value (the orchestrator's grid axes are
# alpha and beta only).
#
# Give each concurrent GPU its own LOG_DIR: run_train_grid_seed.py
# read-modify-writes summary.json there.
set -u

WAIT_PID=${WAIT_PID:-0}
GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_1hr}
ALPHAS=${ALPHAS:?set ALPHAS}
BETAS=${BETAS:?set BETAS}
SEEDS=${SEEDS:-"0 1 2"}
LOG_DIR=${LOG_DIR:?set LOG_DIR}
export ASAP_EPS=${ASAP_EPS:-1e-5}

cd "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$WAIT_PID" != "0" ]; then
    echo "[queue-asap $PROFILE] waiting for pid $WAIT_PID to release GPU$GPU ..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[queue-asap $PROFILE] pid $WAIT_PID exited at $(date -Is)."
    sleep 30
fi

# CWK's setup_path.sh, reached via set_config.sh, expands $PYTHONPATH
# unguarded, which aborts under `set -u` when launched detached (nohup)
# from a shell that never exported one.
export PYTHONPATH="${PYTHONPATH:-}"
source ./set_config.sh
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"

export CUDA_VISIBLE_DEVICES=$GPU
echo "[queue-asap $PROFILE] gpu=$GPU alphas=[$ALPHAS] betas=[$BETAS]" \
     "asap_eps=$ASAP_EPS seeds=[$SEEDS]"
exec python run_train_grid_seed.py \
    run_train_dynamic_grid_asap.sh \
    --alphas $ALPHAS \
    --betas $BETAS \
    --seeds $SEEDS \
    --profile "$PROFILE" \
    --log-dir "$LOG_DIR"
