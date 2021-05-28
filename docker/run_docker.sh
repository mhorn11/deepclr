#!/bin/bash

# check if inside container
tput setaf 1
if [ -n "$DOCKER_MACHINE_NAME" ]; then
  >&2 echo "Error: You probably are already inside a docker container!"
  tput sgr 0
  exit 1
elif [ ! -e /var/run/docker.sock ]; then
  >&2 echo "Error: Either docker is not installed or you are already inside a docker container!"
  tput sgr 0
  exit 1
fi
tput sgr 0

# directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DEEPCLR_DIR="$(readlink -f "${SCRIPT_DIR}/../")"

# default config
CONTAINER_NAME="deepclr"

IMAGE_REGISTRY="docker.pkg.github.com/mhorn11/deepclr/"
IMAGE_NAME="deepclr"
IMAGE_TAG="1.0"

# read arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --deps)
    IMAGE_NAME="deepclr-deps"
    DEPS=1
    shift # past argument
    ;;
    --user)
    USER=1
    shift # past argument
    ;;
    --local)
    IMAGE_REGISTRY=""
    shift # past argument
    ;;
    -m|--mount)
    MOUNT_DIRS+=("$2")
    shift # past argument
    shift # past value
    ;;
    -n|--name)
    CONTAINER_NAME="$2"
    shift # past argument
    shift # past value
    ;;
    --tag)
    IMAGE_TAG="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
    SHOW_HELP=1
    break
    ;;
    *)    # invalid option
    if [[ $1 == -* ]]; then
      echo "Invalid argument '$1'."
      SHOW_HELP=1
      break
    else
      POSITIONAL+=("$1")
      shift # past argument
    fi
    ;;
  esac
done

# pass either all positional arguments or the shell to docker
if [ ${#POSITIONAL[@]} -eq 0 ]; then
  ARGS=${SHELL}
else
  ARGS="${POSITIONAL[@]}"
fi

# show help
if [ "$SHOW_HELP" = 1 ]; then
  echo "Usage: ./run_docker.bash [--deps] [--home] [--local] [-m|--mount DIR [-m|--mount DIR ...]] [--tag TAG] [-h|--help] [ENTRYPOINT]"
  echo ""
  echo "If no ENTRYPOINT is given, your shell ($SHELL) is used."
  echo ""
  echo "Options:"
  echo " * --deps:         Use image with dependencies only (deepclr-deps) instead of full DeepCLR image (deepclr)"
  echo " * --user:         Use current user and group within the container and mount the home directory."
  echo " * --local:        Use local image instead of image from GitHub registry"
  echo " * -m|--mount DIR: Mount directory DIR"
  echo " * -n|--name NAME: Docker container name (default: deepclr)"
  echo " * --tag TAG:      Image tag (default: latest version)"
  echo " * -h|--help:      Show this message"
  echo ""
  exit 1
fi

# docker arguments
DOCKER_ARGS=(
  # mount DeepCLR directory
  -v "${DEEPCLR_DIR}":"${DEEPCLR_DIR}"

  # xserver access for visualization in test scripts
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  -e DISPLAY="${DISPLAY}"

  # container name
  --name "${CONTAINER_NAME}"
  -h "${CONTAINER_NAME}"
  -e CONTAINER_NAME="${CONTAINER_NAME}"
  -e DOCKER_MACHINE_NAME="${CONTAINER_NAME}"

  # misc
  -w "${DEEPCLR_DIR}"  # set DeepCLR directory as workspace
  -it  # run container in interactive mode
  --rm  # automatically remove container when it exits
  --ulimit nofile=1024  # makes forking processes faster, see https://github.com/docker/for-linux/issues/502
)

# dependency vs deploy image
if [ "${DEPS}" = 1 ]; then
  DOCKER_ARGS+=(-e PYTHONPATH="${DEEPCLR_DIR}:${PYTHONPATH}")
else
  DOCKER_ARGS+=(-e PYTHONPATH="${PYTHONPATH}")
fi

# environment variables
if [[ -n "${KITTI_PATH}" ]]; then
  DOCKER_ARGS+=(-e KITTI_PATH="${KITTI_PATH}")
fi
if [[ -n "${MODELNET40_PATH}" ]]; then
  DOCKER_ARGS+=(-e MODELNET40_PATH="${MODELNET40_PATH}")
fi
if [[ -n "${MODEL_PATH}" ]]; then
  DOCKER_ARGS+=(-e MODEL_PATH="${MODEL_PATH}")
fi

# user
if [ "${USER}" = 1 ]; then
  DOCKER_ARGS+=(
    -v /etc/passwd:/etc/passwd:ro
    -v /etc/group:/etc/group:ro
    --user "$(id -u):$(id -g)"
    -v "${HOME}:${HOME}"
  )
else
  DOCKER_ARGS+=(
    --user 10000:10001
  )
fi

# mount directories
for dir in "${MOUNT_DIRS[@]}"; do
  DOCKER_ARGS+=(-v "${dir}:${dir}")
done

# run container
docker run --gpus all \
  "${DOCKER_ARGS[@]}" \
  "${IMAGE_REGISTRY}${IMAGE_NAME}:${IMAGE_TAG}" \
  "${ARGS}" || exit 1
