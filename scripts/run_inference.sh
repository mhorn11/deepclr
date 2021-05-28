#!/bin/bash

# read arguments
ARGS=("$@")

if [[ $# -eq 1 ]]; then
  OUTPUT_DIR=${ARGS[0]}
else
  echo "Usage: ./run_inference.sh OUTPUT_DIR"
  exit 1
fi

# directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCENARIO_DIR="$(readlink -f "${SCRIPT_DIR}/../configs/scenarios")"

# config
declare -A MODELS
MODELS["kitti_04_10"]="kitti_00-03_05-09"
MODELS["kitti_07-10"]="kitti_00-06"
MODELS["kitti_00-10"]="kitti_00-10"
MODELS["kitti_11-21"]="kitti_00-10"
MODELS["kitti_pairs"]="kitti_pairs"
MODELS["modelnet40_unseen"]="modelnet40"
MODELS["modelnet40_seen"]="modelnet40"

# command
CMD="${SCRIPT_DIR}/inference.py"

# iterate
for scenario in "${!MODELS[@]}"; do
  model="${MODELS[$scenario]}"
  "${CMD}" "${SCENARIO_DIR}/${scenario}.yaml" "${model}" "${OUTPUT_DIR}" || exit 1
done
