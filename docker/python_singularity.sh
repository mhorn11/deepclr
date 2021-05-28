#!/bin/bash

# read arguments
ARGS=("$@")
if [[ $# -ge 2 ]]; then
  SINGULARITY_IMAGE=${ARGS[0]}
  PYTHON_ARGS=${ARGS[*]:1}
else
  echo "Usage: ./python_singularity.sh SINGULARITY_IMAGE PYTHON_ARGS [PYTHON_ARGS [...]]"
  exit 1
fi

# directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DEEPCLR_DIR="$(readlink -f "${SCRIPT_DIR}/../")"

# singularity arguments
SINGULARITY_ARGS=(
  --nv  # GPU support
  --env PYTHONPATH="${DEEPCLR_DIR}:${PYTHONPATH}"
)

# environment variables
if [[ -n "${KITTI_PATH}" ]]; then
  SINGULARITY_ARGS+=(--env KITTI_PATH="${KITTI_PATH}")
fi
if [[ -n "${MODELNET40_PATH}" ]]; then
  SINGULARITY_ARGS+=(--env MODELNET40_PATH="${MODELNET40_PATH}")
fi
if [[ -n "${MODEL_PATH}" ]]; then
  SINGULARITY_ARGS+=(--env MODEL_PATH="${MODEL_PATH}")
fi

# run singularity
singularity exec "${SINGULARITY_ARGS[@]}" "${SINGULARITY_IMAGE}" python ${PYTHON_ARGS[@]}
