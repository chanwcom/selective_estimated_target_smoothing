#!/bin/bash
# Decodes every 15000-step WSD checkpoint as it appears, on CPU only.
#
# Training owns GPU 0 and 1 and must never wait for anything here, so this
# never touches a GPU: CUDA_VISIBLE_DEVICES is emptied, --device cpu is
# passed, and thread counts are capped so the box's 80 cores stay ahead of
# the four training dataloaders (~1.6 cores each).
#
# beam search was already CPU-side (lexicon-free flashlight, no LM); only
# the encoder forward moves off the GPU, and it runs while the next seed
# trains rather than after it.
#
# Scheduling note: the loop used to end in `wait`, which blocks until every
# job launched in that pass has finished all four of its decodes. A lane
# that finishes a checkpoint a few minutes before the others therefore got
# decoded alone while PAR-1 slots sat idle, and the checkpoints that landed
# during that window were not even discovered until it was done -- measured
# at 21.5 min per decode, that serialises about four hours of work that
# fits in one. Rescanning on a timer instead keeps PAR jobs in flight.
#
# Rescanning needs its own guard, though: the decode.jsonl test only turns
# true once the LAST of a tag's four rows is written, so a rescan would
# relaunch a tag still in flight. $STARTED records what this process has
# launched. It is cleared at startup, which is what makes a restart resume
# correctly: decode() skips each (tag, split, decoder) already in
# decode.jsonl, so a relaunched tag redoes only its missing combinations.
#
# None of this touches a decode: same binary, same flags, same checkpoint,
# and OMP_NUM_THREADS is per process, so a job's thread count -- and its
# reduction order -- does not depend on how many jobs run beside it.
set -u
cd /mnt/synology_nas_00/chanwcom/local_repository/selective_estimated_target_smoothing
export PYTHONPATH="${PYTHONPATH:-}"; source ./set_config.sh >/dev/null 2>&1
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
OUT=inference_logs_wsd
N=/mnt/synology_nas_00/chanwcom/models
PAR=${PAR:-4}
STARTED=$OUT/.started
touch $OUT/decode.jsonl
# KEEP_STARTED=1 preserves the file across a restart, so a tag whose decode
# was orphaned by killing an older watcher is not picked up a second time.
[ "${KEEP_STARTED:-0}" = "1" ] || : > $STARTED

decode() {
  local ck="$1" tag="$2"
  for split in test-clean test-other; do
    for dec in pipeline beam_search; do
      grep -q "\"$tag\",\"split\":\"$split\",\"decoder\":\"$dec\"" $OUT/decode.jsonl && continue
      w=$(python wav2vec2_inference.py --checkpoint_dir "$ck" --vocab_size 32 \
            --decoder $dec --test_split $split --device cpu 2>/dev/null \
          | grep -oE "'wer': [0-9.]+" | tail -1 | grep -oE "[0-9.]+")
      [ -z "$w" ] && { echo "  [$(date +%H:%M:%S)] $tag $split $dec FAILED"; continue; }
      echo "{\"tag\":\"$tag\",\"split\":\"$split\",\"decoder\":\"$dec\",\"beam\":40,\"wer\":$w}" >> $OUT/decode.jsonl
      echo "  [$(date +%H:%M:%S)] $tag $split $dec -> $w"
    done
  done
}

while true; do
  # Both machines write under $N/<hostname>; u22 is a symlink to
  # slpl_4090_00, so -type d (which does not follow links) keeps the same
  # directory from being scanned twice. Neither machine runs its own
  # watcher, so this one covers seeds 0-2 (u24) and 3-4 (slpl_4090_00).
  for d in $(find $N -maxdepth 1 -mindepth 1 -type d \
               -exec sh -c 'ls -d "$1"/*100hr_wsd_shc_15000steps*/checkpoint-15000 2>/dev/null' _ {} \;); do
    [ -d "$d" ] || continue
    nm=$(basename $(dirname $d))
    a=$(echo $nm|grep -oE 'alpha_[0-9p]+'|sed 's/alpha_//;s/p/./')
    b=$(echo $nm|grep -oE 'beta_[0-9p]+'|sed 's/beta_//;s/p/./')
    s=$(echo $nm|grep -oE 'seed[0-9]')
    m=$([ "$b" = "1.0" ] && echo LS || echo FAS)
    host=$(echo "$d" | sed "s|$N/||; s|/.*||")
    tag="${m}_a${a}_${s}_${host}"
    grep -qxF "$tag" $STARTED && continue
    grep -q "\"$tag\",\"split\":\"test-other\",\"decoder\":\"beam_search\"" $OUT/decode.jsonl && continue
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
    echo "$tag" >> $STARTED
    echo "=== $(date +%H:%M:%S) start $tag  (in flight: $(( $(jobs -rp | wc -l) + 1 ))/$PAR)"
    decode "$d" "$tag" &
  done
  sleep 60
done
