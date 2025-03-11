#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)

# When executed on homebrew installed before it got M1/arm64 support
# uname -m can report x86_64 for Apple Silicon M1 (arm64)
# Uninstall Homebrew and start over
# https://github.com/homebrew/install#uninstall-homebrew
ARCH=$(uname -m)
OS_NAME=$(uname -s)
echo "Running tests in a Docker container on ${OS_NAME} ${ARCH}"

if [ "$ARCH" == "aarch64" ] || [ "$ARCH" == "arm64" ]; then
  MINIFORG_FILENAME="Miniforge3-Linux-aarch64.sh"
else
  MINIFORG_FILENAME="Miniforge3-Linux-x86_64.sh"
fi

IMAGE_TAG=mlflow-test-env
CONTAINER_NAME="mlflow-test"

DOCKER_BUILDKIT=1 \
docker build \
  -f "${ROOT_DIR}/dev/Dockerfile.test" \
  -t ${IMAGE_TAG} \
  "$ROOT_DIR" \
  --build-arg MINIFORG_FILENAME="$MINIFORG_FILENAME"
docker container rm -f ${CONTAINER_NAME} 2>/dev/null
docker run \
  -v "${ROOT_DIR}"/tests:/app/tests \
  -v "${ROOT_DIR}"/mlflow:/app/mlflow \
  -v "${ROOT_DIR}"/pylintrc:/app/pylintrc \
  -v "${ROOT_DIR}"/pyproject.toml:/app/pyproject.toml \
  -v "${ROOT_DIR}"/pytest.ini:/app/pytest.ini \
  -v "${ROOT_DIR}"/conftest.py:/app/conftest.py \
  --name ${CONTAINER_NAME} \
  -it ${IMAGE_TAG}

docker container rm ${CONTAINER_NAME}
