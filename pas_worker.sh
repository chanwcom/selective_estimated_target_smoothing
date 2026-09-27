#!/bin/bash
# pas_worker.sh <gpu> [slot_label]
#
# One worker = one run at a time on one GPU. Start as many per GPU as the
# card holds (two on a 5090, one on a 4090). Each pulls the next job from
# the queue, runs it, and takes the next one -- whatever happened to the
# last.
#
# Five ways a sweep has stalled here before, and what stops each:
#
#   * one death killing the rest -- workers share nothing but the queue
#     file, never use `set -e`, and treat a non-zero exit as "record and
#     continue".
#   * a slot sitting idle with jobs waiting -- a worker stops only when the
#     queue is empty.
#   * a run hung forever without dying. On 2026-09-26 an RNN-T run sat at
#     step 0 for two days at 0.1% CPU holding 20 GiB while its chain script
#     waited. The watchdog kills a run whose log has not grown in
#     STALL_MIN minutes.
#   * a job vanishing because the worker died while holding it. Claims are
#     recorded in-flight; a starting worker requeues anything whose owner
#     is gone, so no job is silently lost.
#   * a broken environment burning the whole queue in seconds, marking
#     every cell FAILED. Three consecutive fast failures stop the worker:
#     that is a machine problem, not a cell problem, and continuing
#     destroys the queue.
#
# Queue is plain text, one job per line: <method> <loss> <alpha> <seed>
# Claims are atomic under flock, so two workers never take the same job.
set -u

GPU=$1
SLOT=${2:-$$}
REPO=/mnt/synology_nas_00/chanwcom/local_repository/selective_estimated_target_smoothing
QUEUE=${QUEUE:-$REPO/pas_queue.txt}
INFLIGHT=${INFLIGHT:-$REPO/pas_inflight.tsv}
STATUS=${STATUS:-/mnt/synology_nas_00/chanwcom/logs/pas_queue_status.tsv}
RESULTS=${RESULTS:-/mnt/synology_nas_00/chanwcom/results/RESULT_PAS.md}
LOGS=${LOGS:-/mnt/synology_nas_00/chanwcom/logs}
STALL_MIN=${STALL_MIN:-25}
FAST_FAIL_SEC=${FAST_FAIL_SEC:-120}
MAX_FAST_FAILS=${MAX_FAST_FAILS:-3}
RUNNER=${RUNNER:-$REPO/run_pas.sh}
LOCK=$QUEUE.lock

say() { echo "[gpu$GPU/$SLOT $(date +%H:%M:%S)] $*"; }

touch "$QUEUE" "$INFLIGHT"
[ -s "$STATUS" ] || printf 'finished_at\trun\tstatus\tseconds\tnote\tgpu\n' > "$STATUS"

# Put back anything claimed by a worker that is no longer alive. Without
# this a killed worker takes its job to the grave and the cell is missing
# from the results with nothing to say why.
reclaim_orphans() {
    exec 9>"$LOCK"; flock 9
    local back=0 line pid job
    : > "$INFLIGHT.keep"
    while IFS=$'\t' read -r pid job; do
        [ -z "${pid:-}" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            printf '%s\t%s\n' "$pid" "$job" >> "$INFLIGHT.keep"
        else
            printf '%s\n' "$job" >> "$QUEUE"
            back=$((back+1))
        fi
    done < "$INFLIGHT"
    mv "$INFLIGHT.keep" "$INFLIGHT"
    flock -u 9; exec 9>&-
    [ "$back" -gt 0 ] && say "requeued $back job(s) orphaned by a dead worker"
    return 0
}

claim() {
    exec 9>"$LOCK"; flock 9
    local job=""
    if [ -s "$QUEUE" ]; then
        job=$(head -1 "$QUEUE")
        tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
        printf '%s\t%s\n' "$$" "$job" >> "$INFLIGHT"
    fi
    flock -u 9; exec 9>&-
    printf '%s' "$job"
}

release() {   # drop our in-flight row once the job is accounted for
    exec 9>"$LOCK"; flock 9
    grep -v -P "^$$\t" "$INFLIGHT" > "$INFLIGHT.tmp" 2>/dev/null || : > "$INFLIGHT.tmp"
    mv "$INFLIGHT.tmp" "$INFLIGHT"
    flock -u 9; exec 9>&-
}

record() {   # run status seconds note
    exec 8>>"$STATUS.lock"; flock 8
    printf '%s\t%s\t%s\t%s\t%s\tgpu%s\n' "$(date -Is)" "$1" "$2" "$3" "$4" "$GPU" >> "$STATUS"
    flock -u 8; exec 8>&-
}

trap 'say "interrupted -- releasing claim"; release; exit 130' INT TERM

reclaim_orphans
FAST_FAILS=0

while true; do
    JOB=$(claim)
    if [ -z "$JOB" ]; then
        say "queue empty -- stopping"
        break
    fi
    set -- $JOB
    METHOD=$1; LOSS=$2; ALPHA=$3; SEED=$4
    if [ "$METHOD" = "baseline" ]; then
        NAME=baseline_${LOSS}_libri100hr_s${SEED}
    else
        NAME=${METHOD}_${LOSS}_libri100hr_alpha_$(echo "$ALPHA" | tr '.' 'p')_s${SEED}
    fi
    LOG=$LOGS/$NAME.log

    # A run that already produced a result row is done. This is what makes
    # a worker restartable mid-sweep instead of redoing finished work.
    if grep -qs "chanwcom/models/$NAME\`" "$RESULTS"; then
        say "$NAME already has a result row -- skipping"
        record "$NAME" "skipped" 0 "result row exists"
        release; continue
    fi

    say "starting $NAME"
    T0=$(date +%s)
    bash "$RUNNER" "$METHOD" "$LOSS" "$ALPHA" "$SEED" "$GPU" &
    RUN_PID=$!

    STALLED=no
    while kill -0 "$RUN_PID" 2>/dev/null; do
        sleep 20
        if [ -f "$LOG" ]; then
            AGE=$(( ( $(date +%s) - $(stat -c %Y "$LOG") ) / 60 ))
            if [ "$AGE" -ge "$STALL_MIN" ]; then
                say "$NAME: log idle ${AGE}m >= ${STALL_MIN}m -- killing as hung"
                STALLED=yes
                pkill -9 -P "$RUN_PID" 2>/dev/null
                kill -9 "$RUN_PID" 2>/dev/null
                break
            fi
        fi
    done
    wait "$RUN_PID" 2>/dev/null; RC=$?
    SECS=$(( $(date +%s) - T0 ))
    release

    if [ "$STALLED" = yes ]; then
        record "$NAME" "HUNG" "$SECS" "log idle >= ${STALL_MIN}m, killed"
        say "$NAME HUNG after ${SECS}s -- next job"
        FAST_FAILS=0
    elif [ "$RC" -ne 0 ]; then
        record "$NAME" "FAILED" "$SECS" "rc=$RC"
        say "$NAME FAILED rc=$RC after ${SECS}s -- next job"
        if [ "$SECS" -lt "$FAST_FAIL_SEC" ]; then
            FAST_FAILS=$((FAST_FAILS+1))
            if [ "$FAST_FAILS" -ge "$MAX_FAST_FAILS" ]; then
                say "STOPPING: $FAST_FAILS jobs failed in under ${FAST_FAIL_SEC}s each."
                say "That is the machine, not the cells. Queue left intact for a rerun."
                record "worker_gpu${GPU}_${SLOT}" "WORKER_STOPPED" 0 \
                       "$FAST_FAILS fast failures -- environment suspected"
                break
            fi
        else
            FAST_FAILS=0
        fi
    else
        record "$NAME" "ok" "$SECS" ""
        say "$NAME ok in ${SECS}s"
        FAST_FAILS=0
    fi
done
