#!/usr/bin/env bash

set -ex

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
conda create -q -n test-environment python=3.5
source activate test-environment

python --version
pip install --upgrade pip==19.3.1

# Install Python test dependencies only if we're running Python tests
if [[ "$INSTALL_SMALL_PYTHON_DEPS" == "true" ]]; then
  pip install -r ./travis/small-requirements.txt
fi
if [[ "$INSTALL_LARGE_PYTHON_DEPS" == "true" ]]; then
  pip install -r ./travis/large-requirements.txt
  # Hack: make sure all spark-* scripts are executable. 
  # Conda installs 2 version spark-* scripts and makes the ones spark
  # uses not executable. This is a temporary fix to unblock the tests.
  ls -lha $(find $CONDA_DIR/envs/test-environment/ -path "*bin/spark-*")
  chmod 777 $(find $CONDA_DIR/envs/test-environment/ -path "*bin/spark-*")
  ls -lha $(find $CONDA_DIR/envs/test-environment/ -path "*bin/spark-*")
fi

pip install .
export MLFLOW_HOME=$(pwd)

# Print current environment info
pip list
which mlflow
echo $MLFLOW_HOME

# Turn off trace output & exit-on-errors
set +ex
