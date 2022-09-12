#!/usr/bin/env bash

function error_handling() {
  echo "Error occurred in dev-env-setup.sh script at line: ${1}"
  echo "Line exited with status: ${2}"
}

trap 'error_handling ${LINENO} $?' ERR

set -o errexit
set -o errtrace
set -o pipefail
shopt -s inherit_errexit
set -x

err=0

MLFLOW_HOME=$(pwd)
export MLFLOW_HOME

# Run the installation of the environment
DEV_DIR=$MLFLOW_HOME/.venvs/mlflow-dev

"$MLFLOW_HOME"/dev/dev-env-setup.sh -d "$DEV_DIR" -f

source "$DEV_DIR/bin/activate"

# Check that packages are installed

SKLEARN_VER=$(pip freeze | grep "scikit-learn")

if [ -z "$SKLEARN_VER" ]; then
  err=$((err + 1))
fi

min_py_version="3.8"
installed_py_version=$(python --version | grep -E -o "([0-9]{1,}\.[0-9]{1,})")

# shellcheck disable=SC2053
if [[ $min_py_version != $installed_py_version ]]; then
  err=$((err + 1))
fi

test $err = 0