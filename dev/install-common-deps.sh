#!/usr/bin/env bash

set -ex

function retry-with-backoff() {
    for BACKOFF in 0 1 2 4 8 16 32 64; do
        sleep $BACKOFF
        if "$@"; then
            return 0
        fi
    done
    return 1
}

# Cleanup apt repository to make room for tests.
sudo apt clean
df -h

# Miniconda is pre-installed in the virtual-environments for GitHub Actions.
# See this repository: https://github.com/actions/virtual-environments
CONDA_DIR=/usr/share/miniconda
export PATH="$CONDA_DIR/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
# Useful for debugging any issues with conda
conda info -a
conda create -q -n test-environment python=3.6
source activate test-environment

python --version
pip install --upgrade pip==19.3.1

if [[ "$MLFLOW_SKINNY" == "true" ]]; then
  pip install . --upgrade
else
  pip install .[extras] --upgrade
fi
export MLFLOW_HOME=$(pwd)

# Install Python test dependencies only if we're running Python tests
if [[ "$INSTALL_SMALL_PYTHON_DEPS" == "true" ]]; then
  # When downloading large packages from PyPI, the connection is sometimes aborted by the
  # remote host. See https://github.com/pypa/pip/issues/8510.
  # As a workaround, we retry installation of large packages.
  retry-with-backoff pip install --quiet -r ./dev/small-requirements.txt
fi
if [[ "$INSTALL_SKINNY_PYTHON_DEPS" == "true" ]]; then
  retry-with-backoff pip install --quiet -r ./dev/skinny-requirements.txt
fi
if [[ "$INSTALL_LARGE_PYTHON_DEPS" == "true" ]]; then
  retry-with-backoff pip install --quiet -r ./dev/large-requirements.txt
  retry-with-backoff pip install --quiet -r ./dev/extra-ml-requirements.txt
  # Hack: make sure all spark-* scripts are executable.
  # Conda installs 2 version spark-* scripts and makes the ones spark
  # uses not executable. This is a temporary fix to unblock the tests.
  ls -lha $(find $CONDA_DIR/envs/test-environment/ -path "*bin/spark-*")
  chmod 777 $(find $CONDA_DIR/envs/test-environment/ -path "*bin/spark-*")
  ls -lha $(find $CONDA_DIR/envs/test-environment/ -path "*bin/spark-*")
fi

# Print current environment info
pip list
which mlflow
echo $MLFLOW_HOME

# Turn off trace output & exit-on-errors
set +ex
