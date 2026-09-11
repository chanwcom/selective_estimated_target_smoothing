#!/bin/bash
# Usage: bash queue_active_support_all.sh
#
# Full AS sweep: 4 alphas x 3 seeds x {libri_light_1hr, libri_light_10hr}
# = 24 runs, one alpha pinned per GPU so all four run in parallel.
#
# 1hr runs first on every GPU, then 10hr, so the (faster, 2000-step) 1hr
# table is complete and readable well before the 10hr one finishes.
#
# Each (profile, alpha) gets its own LOG_DIR because run_train_grid_seed.py
# read-modify-writes summary.json there; concurrent sweeps sharing a
# directory would clobber each other's cells.
set -u

ALPHAS=${ALPHAS:-"0.05 0.1 0.15 0.2"}
GPUS=${GPUS:-"0 1 2 3"}
SEEDS=${SEEDS:-"0 1 2"}

cd "$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -r -a alpha_arr <<< "$ALPHAS"
read -r -a gpu_arr <<< "$GPUS"

if [ "${#alpha_arr[@]}" -ne "${#gpu_arr[@]}" ]; then
    echo "ERROR: ${#alpha_arr[@]} alphas but ${#gpu_arr[@]} GPUs; need one each." >&2
    exit 1
fi

for i in "${!alpha_arr[@]}"; do
    alpha=${alpha_arr[$i]}
    gpu=${gpu_arr[$i]}
    tag=$(echo "$alpha" | tr . p)
    for profile in libri_light_1hr libri_light_10hr; do
        short=${profile#libri_light_}
        log_dir="grid_logs_${short}_as_a${tag}"
        mkdir -p "$log_dir"
        echo "GPU$gpu alpha=$alpha $profile -> $log_dir"
    done
done

for i in "${!alpha_arr[@]}"; do
    alpha=${alpha_arr[$i]}
    gpu=${gpu_arr[$i]}
    tag=$(echo "$alpha" | tr . p)
    nohup bash -c "
        GPU=$gpu ALPHAS='$alpha' SEEDS='$SEEDS' PROFILE=libri_light_1hr \
            LOG_DIR=grid_logs_1hr_as_a$tag bash queue_active_support.sh
        GPU=$gpu ALPHAS='$alpha' SEEDS='$SEEDS' PROFILE=libri_light_10hr \
            LOG_DIR=grid_logs_10hr_as_a$tag bash queue_active_support.sh
    " > "as_sweep_gpu${gpu}_a${tag}.log" 2>&1 &
    echo "launched alpha=$alpha on GPU$gpu (pid $!)"
    sleep 3
done

wait
