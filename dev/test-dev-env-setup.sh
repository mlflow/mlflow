#!/usr/bin/env bash

# Test that the dev-env-setup.sh script installs appropriate versions

set -x

err=0
trap 'err=1' ERR
MLFLOW_HOME=$(pwd)
export MLFLOW_HOME
export MLFLOW_DEV_ENV_REPLACE_ENV=0
export MLFLOW_DEV_ENV_PYENV_INSTALL=1
# Run the installation of the environment
DEV_DIR=$MLFLOW_HOME/.venvs/mlflow-dev

"$MLFLOW_HOME"/dev/dev-env-setup.sh -d "$DEV_DIR" -f -v

# Check that packages are installed

SKLEARN_VER=$(pip freeze | grep "scikit-learn")

if [ -z $SKLEARN_VER ]; then
  err=$((err + 1))
fi

min_py_version=$(grep "python_requires=" "$MLFLOW_HOME/setup.py" | grep -E -o "([0-9]{1,}\.)+[0-9]{1,}")
installed_py_version=$(python --version | grep -E -o "([0-9]{1,}\.[0-9]{1,})")

# shellcheck disable=SC2053
if [[ $min_py_version != $installed_py_version ]]; then
  err=$((err + 1))
fi

test $err = 0
