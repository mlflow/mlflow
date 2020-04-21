#!/usr/bin/env bash

set -ex

# Cleanup apt repository to make room for tests.
sudo apt clean
df -h

sudo mkdir -p /travis-install
# GITHUB_WORKFLOW is set by default during GitHub workflows
if [[ -z $GITHUB_WORKFLOW ]]; then
  sudo chown travis /travis-install
fi
# (The conda installation steps below are taken from http://conda.pydata.org/docs/travis.html)
# We do this conditionally because it saves us some downloading if the
# version is the same.
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh

bash $HOME/miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
# Useful for debugging any issues with conda
conda info -a
if [[ -n "$TRAVIS_PYTHON_VERSION" ]]; then
  conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION
else
  conda create -q -n test-environment python=3.6
fi
source activate test-environment
python --version
pip install --upgrade pip==19.3.1
# Install Python test dependencies only if we're running Python tests
if [[ "$INSTALL_SMALL_PYTHON_DEPS" == "true" ]]; then
  pip install --quiet -r ./travis/small-requirements.txt
fi
if [[ "$INSTALL_LARGE_PYTHON_DEPS" == "true" ]]; then
  pip install --quiet -r ./travis/large-requirements.txt
  # Hack: make sure all spark-* scripts are executable. 
  # Conda installs 2 version spark-* scripts and makes the ones spark
  # uses not executable. This is a temporary fix to unblock the tests.
  ls -lha $(find $HOME/miniconda/envs/test-environment/ -path "*bin/spark-*")
  chmod 777 $(find $HOME/miniconda/envs/test-environment/ -path "*bin/spark-*")
  ls -lha $(find $HOME/miniconda/envs/test-environment/ -path "*bin/spark-*")
fi
pip install .
export MLFLOW_HOME=$(pwd)
# Remove boto config present in Travis VMs (https://github.com/travis-ci/travis-ci/issues/7940)
if [[ -z $GITHUB_WORKFLOW ]]; then
  sudo rm -f /etc/boto.cfg
fi
# Print current environment info
pip list
which mlflow
echo $MLFLOW_HOME

# Turn off trace output & exit-on-errors
set +ex
