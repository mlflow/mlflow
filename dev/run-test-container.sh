#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)
ARCH=$(uname -m)
echo "Running tests in a Docker container on $ARCH"

if [ "$ARCH" == "aarch64" ] || [ "$ARCH" == "arm64" ]; then
  MINIFORG_FILENAME="Miniforge3-Linux-aarch64.sh"
else
  MINIFORG_FILENAME="Miniforge3-Linux-x86_64.sh"
fi

DOCKER_BUILDKIT=1 docker build -f "${ROOT_DIR}/dev/Dockerfile.test" -t mlflow-test-env "$ROOT_DIR" --build-arg MINIFORG_FILENAME="$MINIFORG_FILENAME"
docker container rm -f mlflow-test 2>/dev/null
docker run \
  -v "${ROOT_DIR}"/tests:/app/tests \
  -v "${ROOT_DIR}"/mlflow:/app/mlflow \
  -v "${ROOT_DIR}"/pylintrc:/app/pylintrc \
  -v "${ROOT_DIR}"/pyproject.toml:/app/pyproject.toml \
  -v "${ROOT_DIR}"/pytest.ini:/app/pytest.ini \
  -v "${ROOT_DIR}"/conftest.py:/app/conftest.py \
  --name "mlflow-test" \
  -it mlflow-test-env

docker container rm mlflow-test
