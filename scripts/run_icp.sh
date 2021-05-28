#!/bin/bash

# read arguments
ARGS=("$@")

if [[ $# -eq 1 ]]; then
  OUTPUT_DIR=${ARGS[0]}
else
  echo "Usage: ./run_icp.sh OUTPUT_DIR"
  exit 1
fi

# directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCENARIO_DIR="$(readlink -f "${SCRIPT_DIR}/../configs/scenarios")"

# config
KITTI_PARAMS="--max-distance 10.0 --neighbor-radius 1.0 --max-nn 30"
MODELNET40_PARAMS="--max-distance 0.2 --neighbor-radius 0.2 --max-nn 30"

declare -A SCENARIOS
SCENARIOS["kitti_04_10"]="${KITTI_PARAMS}"
SCENARIOS["kitti_07-10"]="${KITTI_PARAMS}"
SCENARIOS["kitti_pairs"]="${KITTI_PARAMS}"
SCENARIOS["modelnet40_unseen"]="${MODELNET40_PARAMS}"
SCENARIOS["modelnet40_seen"]="${MODELNET40_PARAMS}"

ICP_ALGORITHMS=( "ICP_PO2PO" "ICP_PO2PL" "GICP" )

# command
CMD="${SCRIPT_DIR}/icp.py"

# iterate
for scenario in "${!SCENARIOS[@]}"; do
  params="${SCENARIOS[$scenario]}"
  for algorithm in "${ICP_ALGORITHMS[@]}"; do
    "${CMD}" "${SCENARIO_DIR}/${scenario}.yaml" "${algorithm}" "${OUTPUT_DIR}" ${params} || exit 1
  done
done
