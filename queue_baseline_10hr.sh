#!/bin/bash
# 10hr no-smoothing baseline. alpha=0 means ShcLoss never calls the
# smoothing at all (smoothing_enabled = ... or alpha > 0.0), so the
# label/class distinction is moot here and one run serves both.
# The 1hr table has this cell; the 10hr table does not, which leaves every
# 10hr comparison relative to other smoothers rather than to no smoothing.
set -u
WAIT_PID=${WAIT_PID:-0}; GPU=${GPU:-1}
cd "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$WAIT_PID" != "0" ]; then
  echo "[queue-base10] waiting for pid $WAIT_PID ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[queue-base10] pid $WAIT_PID exited at $(date -Is)."; sleep 30
fi
source ./set_config.sh
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"
export CUDA_VISIBLE_DEVICES=$GPU
exec python run_train_grid_seed.py run_train_dynamic_grid.sh \
  --alphas 0.0 --betas 0.0 --seeds 0 1 2 \
  --profile libri_light_10hr --log-dir grid_logs_10hr
