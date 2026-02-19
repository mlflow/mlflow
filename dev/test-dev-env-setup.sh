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

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT

# Run the installation of the environment
DEV_DIR=$REPO_ROOT/.venvs/mlflow-dev

"$REPO_ROOT"/dev/dev-env-setup.sh -d "$DEV_DIR"

source "$DEV_DIR/bin/activate"

# Check that packages are installed

SKLEARN_VER=$(pip freeze | grep "scikit-learn")

if [ -z "$SKLEARN_VER" ]; then
  err=$((err + 1))
fi

min_py_version="3.10"
installed_py_version=$(python --version | grep -E -o "([0-9]{1,}\.[0-9]{1,})")

# shellcheck disable=SC2053
if [[ $min_py_version != $installed_py_version ]]; then
  err=$((err + 1))
fi

test $err = 0
