#!/bin/bash

# read arguments
ARGS=("$@")

VERSION="v3.6.0"
if [[ $# -eq 2 ]] || [[ $# -eq 3 ]]; then
  OUTPUT_DIR=${ARGS[0]}
  IMAGE=${ARGS[1]}
  if [[ $# -eq 3 ]]; then
    VERSION=${ARGS[2]}
  fi
else
  echo "Usage: ./docker2singularity.sh OUTPUT_DIR IMAGE [VERSION]"
  exit 1
fi

# convert image
docker run \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "${OUTPUT_DIR}":/output \
  --privileged -t --rm \
  "quay.io/singularity/docker2singularity:${VERSION}" \
  "${IMAGE}"
