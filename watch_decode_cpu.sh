#!/bin/bash
# Decodes every 10000-step checkpoint as it appears, on CPU only.
#
# Training owns GPU 0 and 1 and must never wait for anything here, so this
# never touches a GPU: CUDA_VISIBLE_DEVICES is emptied, --device cpu is
# passed, and thread counts are capped so the box's 80 cores stay ahead of
# the four training dataloaders (~1.6 cores each).
#
# beam search was already CPU-side (lexicon-free flashlight, no LM); only
# the encoder forward moves off the GPU, and it runs while the next seed
# trains rather than after it.
set -u
cd /mnt/synology_nas_00/chanwcom/local_repository/selective_estimated_target_smoothing
export PYTHONPATH="${PYTHONPATH:-}"; source ./set_config.sh >/dev/null 2>&1
[ -n "${PYTHON_BIN:-}" ] && export PATH="$PYTHON_BIN:$PATH"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
OUT=inference_logs_10k
N=/mnt/synology_nas_00/chanwcom/models
PAR=${PAR:-4}
touch $OUT/decode.jsonl

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
  running=0
  for d in $N/u24/*100hr_shc_10000steps*/checkpoint-10000; do
    [ -d "$d" ] || continue
    nm=$(basename $(dirname $d))
    a=$(echo $nm|grep -oE 'alpha_[0-9p]+'|sed 's/alpha_//;s/p/./')
    b=$(echo $nm|grep -oE 'beta_[0-9p]+'|sed 's/beta_//;s/p/./')
    s=$(echo $nm|grep -oE 'seed[0-9]')
    m=$([ "$b" = "1.0" ] && echo LS || echo FAS)
    tag="${m}_a${a}_${s}"
    grep -q "\"$tag\",\"split\":\"test-other\",\"decoder\":\"beam_search\"" $OUT/decode.jsonl && continue
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
    echo "=== $(date +%H:%M:%S) start $tag"
    decode "$d" "$tag" &
    running=1
  done
  [ "$running" -eq 0 ] && sleep 300 || wait
done
