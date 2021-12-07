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
pip install --upgrade pip wheel
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
  retry-with-backoff pip install -r requirements/small-requirements.txt
fi
if [[ "$INSTALL_SKINNY_PYTHON_DEPS" == "true" ]]; then
  retry-with-backoff pip install -r requirements/skinny-requirements.txt
fi
if [[ "$INSTALL_LARGE_PYTHON_DEPS" == "true" ]]; then
  retry-with-backoff pip install -r requirements/large-requirements.txt

  # Install prophet's dependencies beforehand, otherwise pip would fail to build a wheel for prophet
  if [[ -z "$(pip cache list prophet --format abspath)" ]]; then
    tmp_dir=$(mktemp -d)
    pip download --no-deps --dest $tmp_dir --no-cache-dir prophet
    tar -zxvf $tmp_dir/*.tar.gz -C $tmp_dir
    pip install -r $(find $tmp_dir -name requirements.txt)
    rm -rf $tmp_dir
  fi

  retry-with-backoff pip install -r requirements/extra-ml-requirements.txt
fi

# Install `mlflow-test-plugin` without dependencies
pip install --no-dependencies tests/resources/mlflow-test-plugin

# Print current environment info
python dev/show_package_release_dates.py
which mlflow
echo $MLFLOW_HOME

# Turn off trace output & exit-on-errors
set +ex
