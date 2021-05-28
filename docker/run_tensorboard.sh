#!/bin/bash

# default config
IMAGE="tensorflow/tensorflow"
CONTAINER_NAME="tensorboard"

# read arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -d|--detach)
    DETACH=1
    shift # past argument
    ;;
    -n|--name)
    CONTAINER_NAME=$2
    shift # past argument
    shift # past value
    ;;
    -s|--stop)
    STOP=1
    shift # past argument
    ;;
    --help)
    SHOW_HELP=1
    break
    ;;
    *)  # other arguments
    POSITIONAL+=("$1")
    shift # past argument
    ;;
  esac
done

# stop detached container
if [ "${STOP}" = 1 ]; then
  echo "Stopping TensorBoard container '${CONTAINER_NAME}'"
  docker stop "${CONTAINER_NAME}"
  exit 0
fi

# check positional arguments
if [ ${#POSITIONAL[@]} -ge 1 ]; then
  LOGDIR="${POSITIONAL[0]}"
  TENSORBOARD_ARGS=${POSITIONAL[*]:1}
else
  echo "Missing argument."
  SHOW_HELP=1
fi

# show help
if [ "${SHOW_HELP}" = 1 ]; then
  echo "Usage: ./run_docker.bash LOGDIR [-d|--detach] [-n|--name] [-s|--stop] [--help] [TENSORBOARD_ARGS [...]]"
  echo ""
  echo "The directory LOGDIR contains your TensorBoard logs. All TENSORBOARD_ARGS are forwarded to TensorBoard."
  echo ""
  echo "Options:"
  echo " * -d|--detach: Detach and run container in background"
  echo " * -n|--name:   Container name (default: tensorboard)"
  echo " * -s|--stop:   Stop detached tensorboard container"
  echo " * --help:      Show this message"
  echo ""
  exit 1
fi

# docker arguments
DOCKER_ARGS=(
  # mount log directory
  -v "${LOGDIR}:/logs:ro"

  # container name
  --name "${CONTAINER_NAME}"
  -h "${CONTAINER_NAME}"

  # misc
  --network=host  # don't isolate container network stack from host
  --rm  # automatically remove container when it exits
)

# detach
if [ "${DETACH}" = 1 ]; then
  DOCKER_ARGS+=(-d)
fi

# run container
echo "Starting TensorBoard container '${CONTAINER_NAME}'"
docker run \
  "${DOCKER_ARGS[@]}" \
  "${IMAGE}" \
  tensorboard --logdir /logs ${TENSORBOARD_ARGS[@]} || exit 1
