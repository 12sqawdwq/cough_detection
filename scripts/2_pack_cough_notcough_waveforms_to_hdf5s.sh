#!/bin/bash
# Usage: bash run_pack.sh <audios_root_dir> <workspace_dir> <dataset_name>

AUDIOS_ROOT=${1:-"./data"}
WORKSPACE=${2:-"data/workspace"}
DATASET_NAME=${3:-"cough_notcough"}

python utils/dataset.py pack_waveforms_to_hdf5 \
    --audios_dir "$AUDIOS_ROOT" \
    --workspace_dir "$WORKSPACE" \
    --dataset_name "$DATASET_NAME"

echo "生成的文件路径:"
echo "  Train: $WORKSPACE/hdf5s/waveforms/$DATASET_NAME/${DATASET_NAME}_train.h5"
echo "  Test:  $WORKSPACE/hdf5s/waveforms/$DATASET_NAME/${DATASET_NAME}_test.h5"
