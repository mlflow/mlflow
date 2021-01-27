#!/bin/bash

# Test that mlflow installation does not modify (downgrade/upgrade/uninstall) packages from a
# specific Anaconda distribution.

set -eux

MLFLOW_DIR=$(cd "$(dirname ${BASH_SOURCE[0]})/.."; pwd)

PYTHON_MAJOR_VERSION=$(python -c "import sys; print(sys.version_info[0])")
DEFAULT_ANACONDA_VERSIONS=([2]="anaconda:2020.11" [3]="anaconda3:2020.11")
ANACONDA_VERSION=${1:-${DEFAULT_ANACONDA_VERSIONS["$PYTHON_MAJOR_VERSION"]}}
docker run --rm -v "${MLFLOW_DIR}:/mnt/mlflow" continuumio/${ANACONDA_VERSION} \
  bash /mnt/mlflow/dev/test-anaconda-compatibility-in-docker.sh
