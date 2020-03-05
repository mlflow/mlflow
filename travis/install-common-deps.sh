#!/usr/bin/env bash

set -ex
sudo mkdir -p /travis-install
sudo chown travis /travis-install
# (The conda installation steps below are taken from http://conda.pydata.org/docs/travis.html)
# We do this conditionally because it saves us some downloading if the
# version is the same.
if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O /travis-install/miniconda.sh;
else
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /travis-install/miniconda.sh;
fi

bash /travis-install/miniconda.sh -b -p $HOME/miniconda
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
  pip install -r ./travis/small-requirements.txt
fi
if [[ "$INSTALL_LARGE_PYTHON_DEPS" == "true" ]]; then
  pip install -r ./travis/large-requirements.txt
fi
pip install .
export MLFLOW_HOME=$(pwd)
# Remove boto config present in Travis VMs (https://github.com/travis-ci/travis-ci/issues/7940)
sudo rm -f /etc/boto.cfg
# Print current environment info
pip list
which mlflow
echo $MLFLOW_HOME

# Turn off trace output & exit-on-errors
set +ex
