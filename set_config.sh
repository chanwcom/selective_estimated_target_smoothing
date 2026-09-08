CWK_SCRIPT_HOME=/mnt/synology_nas_00/chanwcom/local_repository/cognitive_workflow_kit/scripts
DEVICE_ID=${DEVICE_ID:-0}

source $CWK_SCRIPT_HOME/setup_path.sh

export NCCL_P2P_DISABLE=1; export NCCL_IB_DISABLE=1; export CUDA_VISIBLE_DEVICES=$DEVICE_ID;
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
