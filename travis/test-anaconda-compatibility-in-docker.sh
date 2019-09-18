#!/bin/bash

# Test that mlflow installation does not modify (downgrade/upgrade/uninstall) packages from a
# specific Anaconda distribution.
# This script should run inside a continuumio/anaconda docker container with mlflow source
# directory mounted at /mnt/mlflow.
# See test-anaconda-compatibility.sh.

set -euo pipefail

. ~/.bashrc

pip freeze > /tmp/before.txt
pip install --upgrade-strategy only-if-needed -e /mnt/mlflow
pip freeze > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt > /tmp/diff.txt || true
if [[ ! -z $(grep "<" /tmp/diff.txt) ]]; then
  echo "MLflow installation modified the Anaconda distribution:" 1>&2
  cat /tmp/diff.txt 1>&2
  exit 1
else
  echo "MLflow installation did not modify the Anaconda distribution."
  exit 0
fi
