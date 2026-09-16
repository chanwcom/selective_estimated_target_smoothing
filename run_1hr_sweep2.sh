#!/bin/bash
# Usage: WORKER=<n> GPU=<n> bash run_1hr_sweep2.sh
#
# Second 1hr sweep: baseline, textbook LS, and FAS, five seeds each, with
# alpha in {0.05, 0.10, 0.15, 0.20, 0.25, 0.30} for the two smoothed
# methods. Baseline has no parameter to sweep, so it is alpha=0 x 5 seeds.
#
# Ordering is SEED-MAJOR -- every config at seed 0, then every config at
# seed 1, and so on -- so a full cross-method comparison exists after the
# first ~3.5 h instead of after the whole sweep. run_train_grid_seed.py
# cannot express this: its loop is `for alpha: for seed:` (line 276), i.e.
# seed-minor, which finishes one alpha across all five seeds before
# starting the next. Hence this launcher, which builds the task list in
# the order we want and hands it to workers through a shared queue.
#
# One training run per GPU: a 1hr run holds ~21 GB, so two would not fit
# in 24 GB. Two workers, one per GPU.
#
# Each task shells out to exactly the same script, with the same argument
# order and the same log-path convention, that run_train_grid_seed.py's
# run_one() uses. So the logs stay readable by the existing analysis
# scripts, and re-running run_train_grid_seed.py afterwards will skip
# every cell and just regenerate summary.json.
#
# Resumable, and reuses the first sweep: a cell whose log already holds
# both an eval_wer and a train_runtime is skipped. That is why this can be
# launched as the full 65-task request while only the 35 genuinely new
# runs execute -- the first sweep already covers baseline and LS
# alpha=0.05..0.25.
set -u

WORKER=${WORKER:-0}
GPU=${GPU:-0}
QUEUE=${QUEUE:-/tmp/claude-1003/sweep2_1hr_queue}
PROFILE=${PROFILE:-libri_light_1hr}
# Separate directories from the first sweep's. That sweep ran the 2000-step
# linear schedule; libri_light_1hr is now 2500-step WSD, so the logs share
# a file name (the name carries the profile, not the schedule) but not a
# comparable number. Checkpoints separate themselves -- max_steps is in the
# run name -- but logs would overwrite, and a reused 2000-step log would be
# silently treated as this sweep's result by the skip check below.
LS_LOG_DIR=${LS_LOG_DIR:-grid_logs_1hr_wsd_ls_4090}
FAS_LOG_DIR=${FAS_LOG_DIR:-grid_logs_1hr_wsd_fas_4090}

cd "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${PYTHONPATH:-}"
source ./set_config.sh
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"
export CUDA_VISIBLE_DEVICES=$GPU
# Without this the Trainer's dict lines sit in a full stdout pipe and the
# log lags thousands of steps behind the run.
export PYTHONUNBUFFERED=1

# Nothing to do about run_train_dynamic_grid_fas.sh's GPU_MEMORY_FRACTION
# default of 0.46 here: wav2vec2_finetuning_sets.py ignores the flag on
# cards under 30 GiB, which is every GPU on this box, so the run goes
# unconstrained and matches the LS runs (their script never passes the
# flag at all).

mkdir -p "$LS_LOG_DIR" "$FAS_LOG_DIR"

ALPHAS="0.05 0.1 0.15 0.2 0.25 0.3"

# _fmt from run_train_grid_seed.py: 0.05 -> 0p05, 0.1 -> 0p1.
fmt() { echo "${1//./p}"; }

tasks=()
for SEED in 0 1 2 3 4; do
    # Baseline first in each round: it is the reference every other cell
    # in that round is read against.
    tasks+=("baseline|0.0|run_train_dynamic_grid_class_space.sh|$LS_LOG_DIR|$SEED")
    for A in $ALPHAS; do
        tasks+=("LS|$A|run_train_dynamic_grid_class_space.sh|$LS_LOG_DIR|$SEED")
    done
    for A in $ALPHAS; do
        tasks+=("FAS|$A|run_train_dynamic_grid_fas.sh|$FAS_LOG_DIR|$SEED")
    done
done

echo "[sweep2 w$WORKER/gpu$GPU] $(date -Is) ${#tasks[@]} tasks, queue $QUEUE"

next_index() {
    local i
    exec 9>"$QUEUE.lock"
    flock 9
    i=$(cat "$QUEUE" 2>/dev/null || echo 0)
    echo $((i + 1)) > "$QUEUE"
    flock -u 9
    exec 9>&-
    echo "$i"
}

ran=0; skipped=0; failed=0
while :; do
    i=$(next_index)
    [ "$i" -lt "${#tasks[@]}" ] || break
    IFS='|' read -r METHOD ALPHA SCRIPT LOG_DIR SEED <<<"${tasks[$i]}"

    LOG="$LOG_DIR/${PROFILE}_alpha$(fmt "$ALPHA")_beta0p0_seed${SEED}.log"

    # Same completeness test run_train_grid_seed.py's try_load_cached uses:
    # a log counts as done only when the run reached the end, which is what
    # train_runtime marks -- eval_wer alone appears at every eval step.
    if [ -f "$LOG" ] && grep -q "'eval_wer'" "$LOG" && grep -q "'train_runtime'" "$LOG"; then
        skipped=$((skipped + 1))
        continue
    fi

    echo "[sweep2 w$WORKER] $(date +%H:%M:%S) start seed=$SEED $METHOD alpha=$ALPHA"
    if bash "$SCRIPT" "$ALPHA" 0.0 "$SEED" "$PROFILE" > "$LOG" 2>&1; then
        ran=$((ran + 1))
        echo "[sweep2 w$WORKER] $(date +%H:%M:%S) ok    seed=$SEED $METHOD alpha=$ALPHA  $(grep -o "'eval_wer': [0-9.]*" "$LOG" | tail -1)"
    else
        failed=$((failed + 1))
        echo "[sweep2 w$WORKER] $(date +%H:%M:%S) FAIL  seed=$SEED $METHOD alpha=$ALPHA -- see $LOG"
    fi
done

echo "[sweep2 w$WORKER/gpu$GPU] $(date -Is) done: ran $ran, skipped $skipped, failed $failed"
