#!/bin/bash
# Usage: PROFILE=libri_light_10hr WORKER=<n> GPU=<n> bash run_sweep2.sh
#
# Baseline, textbook LS, and FAS over one fine-tuning profile: five seeds
# each, with alpha in {0.05, 0.10, 0.15, 0.20, 0.25, 0.30} for the two
# smoothed methods. Baseline has no parameter to sweep, so it is alpha=0 x
# five seeds. 13 configs x 5 seeds = 65 runs.
#
# Profile-generic version of run_1hr_sweep2.sh, which stays on disk only
# until the 1hr sweep it is currently running finishes: bash reads a script
# incrementally as it executes, so editing or deleting a running one can
# make it execute garbage. Same logic, PROFILE and the log directories are
# arguments.
#
# Ordering is SEED-MAJOR -- every config at seed 0, then every config at
# seed 1, and so on -- so a full cross-method comparison exists after the
# first round rather than only at the end, and stopping after any round
# leaves a complete, balanced result at that many seeds.
# run_train_grid_seed.py cannot express this: its loop is `for alpha: for
# seed:`, which finishes one alpha across all five seeds before starting
# the next.
#
# One training run per GPU. A 1hr run peaks around 21 GiB and a 10hr run is
# no smaller, so two do not fit in 24 GB.
#
# Each task shells out to the same script, with the same argument order and
# the same log-path convention, that run_train_grid_seed.py's run_one()
# uses, so the logs stay readable by the analysis scripts and a later
# run_train_grid_seed.py invocation will skip every cell and just write
# summary.json.
#
# Resumable: a cell whose log already holds both an eval_wer and a
# train_runtime is skipped.
set -u

WORKER=${WORKER:-0}
GPU=${GPU:-0}
PROFILE=${PROFILE:-libri_light_10hr}
QUEUE=${QUEUE:-/tmp/claude-1003/sweep2_${PROFILE}_queue}

# Directories are per profile AND per schedule. The 4000-step linear 10hr
# sweep and this 6000-step WSD one produce the same log file names -- the
# name carries the profile, not the schedule -- so sharing a directory
# would overwrite the old results and, worse, let the skip check below
# mistake one for this sweep's.
_short=${PROFILE#libri_light_}
LS_LOG_DIR=${LS_LOG_DIR:-grid_logs_${_short}_wsd_ls_4090}
FAS_LOG_DIR=${FAS_LOG_DIR:-grid_logs_${_short}_wsd_fas_4090}

cd "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${PYTHONPATH:-}"
source ./set_config.sh
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"
export CUDA_VISIBLE_DEVICES=$GPU
# Without this the Trainer's dict lines sit in a full stdout pipe and the
# log lags thousands of steps behind the run.
export PYTHONUNBUFFERED=1

# set_config.sh already defaults GPU_MEMORY_FRACTION to 0 on cards too small
# to share, which is what makes run_train_dynamic_grid_fas.sh's 0.46 default
# harmless here.

mkdir -p "$LS_LOG_DIR" "$FAS_LOG_DIR"

ALPHAS="0.05 0.1 0.15 0.2 0.25 0.3"
SEEDS="0 1 2 3 4"

# _fmt from run_train_grid_seed.py: 0.05 -> 0p05, 0.1 -> 0p1.
fmt() { echo "${1//./p}"; }

tasks=()
for SEED in $SEEDS; do
    # Baseline first in each round: it is the reference every other cell in
    # that round is read against.
    tasks+=("baseline|0.0|run_train_dynamic_grid_class_space.sh|$LS_LOG_DIR|$SEED")
    for A in $ALPHAS; do
        tasks+=("LS|$A|run_train_dynamic_grid_class_space.sh|$LS_LOG_DIR|$SEED")
    done
    for A in $ALPHAS; do
        tasks+=("FAS|$A|run_train_dynamic_grid_fas.sh|$FAS_LOG_DIR|$SEED")
    done
done

echo "[sweep2 $PROFILE w$WORKER/gpu$GPU] $(date -Is) ${#tasks[@]} tasks, queue $QUEUE"

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
    # train_runtime marks the end of a run, while eval_wer appears at every
    # eval step and so cannot tell a finished run from an interrupted one.
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

echo "[sweep2 $PROFILE w$WORKER/gpu$GPU] $(date -Is) done: ran $ran, skipped $skipped, failed $failed"
