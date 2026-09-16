# Per-shell environment setup. Source this (do not execute it) before
# running anything in this repo:
#
#     source set_config.sh
#
# Machine-specific paths are not defined here -- they live in the gitignored
# `config.local.sh`, so this file stays identical on every machine. See
# `config.local.sh.example`.

_repo_home="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$_repo_home/config.local.sh" ]]; then
    echo "ERROR: $_repo_home/config.local.sh not found." >&2
    echo "       cp config.local.sh.example config.local.sh" >&2
    echo "       then edit it for this machine (see README, Setup)." >&2
    return 1
fi
source "$_repo_home/config.local.sh"

DEVICE_ID=${DEVICE_ID:-0}

# Puts the CWK repo's `scripts/` on PYTHONPATH, which is where `common`
# lives. `cwk` itself comes from `pip install -e .` in that repo.
source "$CWK_HOME/scripts/setup_path.sh"

export NCCL_P2P_DISABLE=1; export NCCL_IB_DISABLE=1; export CUDA_VISIBLE_DEVICES=$DEVICE_ID;
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The training launchers (run_train_dynamic_grid_fas.sh, _asap.sh) default
# GPU_MEMORY_FRACTION to 0.46, which is sized for the 32 GB cards where two
# runs share a GPU. A 24 GB card fits exactly one run -- a 1hr run peaks
# around 21 GiB -- so that fraction caps the only run it has to 10.8 GiB and
# it OOMs in the first backward. Default the fraction off on any box whose
# smallest card cannot be shared, so those launchers work unmodified here.
#
# Rule, not a machine-specific value, so this file stays identical
# everywhere: it reads the hardware rather than being edited per host.
# 30 GiB (30720 MiB) rather than 32 because a nominally 32 GB card reports
# ~31.4 GiB once the driver's reservation is out; the cutoff has to sit
# below the nominal figure or it would fire on the cards the flag is for.
#
# Smallest card, not $DEVICE_ID's: the launchers re-point
# CUDA_VISIBLE_DEVICES per worker after sourcing this, so which card a run
# lands on is not known here. Unconstrained is the safe default -- it is
# what a lone run wants -- and wav2vec2_finetuning_sets.py makes the exact
# per-device call anyway, ignoring the flag on whatever card it opens.
#
# An explicit GPU_MEMORY_FRACTION from the caller is left alone.
if [[ -z "${GPU_MEMORY_FRACTION:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    _min_vram_mib=$(nvidia-smi --query-gpu=memory.total \
                    --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
    if [[ -n "$_min_vram_mib" ]] && (( _min_vram_mib < 30720 )); then
        export GPU_MEMORY_FRACTION=0
    fi
    unset _min_vram_mib
fi
