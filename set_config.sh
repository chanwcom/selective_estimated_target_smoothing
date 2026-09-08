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
