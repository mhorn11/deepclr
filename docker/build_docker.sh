#!/bin/bash

# read arguments
ARGS=("$@")

if [[ $# -ge 2 ]]; then
  TYPE=${ARGS[0]}
  TAG=${ARGS[1]}
  DOCKER_ARGS=${ARGS[*]:2}
else
  echo "Usage: ./build_docker.sh {deps, deploy} TAG [DOCKER_ARGS [...]]"
  exit 1
fi

# directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DEEPCLR_DIR="$(readlink -f "${SCRIPT_DIR}/../")"

# type
if [[ "${TYPE}" = "deps" ]]; then
  # config
  IMAGE_NAME="deepclr-deps:${TAG}"
  DOCKERFILE="${SCRIPT_DIR}/dockerfiles/deps.Dockerfile"

  # prepare extern
  "${DEEPCLR_DIR}/extern/prepare.sh"

elif [[ "${TYPE}" = "deploy" ]]; then
  # config
  IMAGE_NAME="deepclr:${TAG}"
  DOCKERFILE="${SCRIPT_DIR}/dockerfiles/deploy.Dockerfile"

else
  echo "Invalid Docker image type: ${TYPE}"
  exit 1
fi

# build
docker build \
  "${DEEPCLR_DIR}" \
  -t "${IMAGE_NAME}" \
  -f "${DOCKERFILE}" \
  --build-arg TAG="${TAG}" \
  ${DOCKER_ARGS[@]}
