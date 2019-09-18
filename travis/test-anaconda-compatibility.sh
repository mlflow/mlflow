#!/bin/bash

# Test that mlflow installation does not modify (downgrade/upgrade/uninstall) packages from a
# specific Anaconda distribution.

set -e

MLFLOW_DIR=$(cd "$(dirname ${BASH_SOURCE[0]})/.."; pwd)

ANACONDA_VERSION=${1:-"anaconda3:2019.03"}
docker run --rm -v "${MLFLOW_DIR}:/mnt/mlflow" continuumio/${ANACONDA_VERSION} \
  bash /mnt/mlflow/travis/test-anaconda-compatibility-in-docker.sh
