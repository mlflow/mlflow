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

python --version
pip install --upgrade pip
pip --version

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
  retry-with-backoff pip install -r ./dev/small-requirements.txt
fi
if [[ "$INSTALL_SKINNY_PYTHON_DEPS" == "true" ]]; then
  retry-with-backoff pip install -r ./dev/skinny-requirements.txt
fi
if [[ "$INSTALL_LARGE_PYTHON_DEPS" == "true" ]]; then
  retry-with-backoff pip install -r ./dev/large-requirements.txt
  retry-with-backoff pip install -r ./dev/extra-ml-requirements.txt
  # Hack: make sure all spark-* scripts are executable.
  # Conda installs 2 version spark-* scripts and makes the ones spark
  # uses not executable. This is a temporary fix to unblock the tests.
  SITE_PACKAGES_DIR=$(python -c "import site; print (site.getsitepackages())[0]")
  ls -lha $(find $SITE_PACKAGES_DIR -path "*bin/spark-*")
  chmod 777 $(find $SITE_PACKAGES_DIR -path "*bin/spark-*")
  ls -lha $(find $SITE_PACKAGES_DIR -path "*bin/spark-*")
fi

# Install `mlflow-test-plugin` without dependencies
pip install --no-dependencies tests/resources/mlflow-test-plugin

# Print current environment info
pip list
which mlflow
echo $MLFLOW_HOME

# Turn off trace output & exit-on-errors
set +ex
