#!/usr/bin/env bash

set -ex

function retry-with-backoff() {
    for BACKOFF in 0 1 2; do
        sleep $BACKOFF
        if "$@"; then
            return 0
        fi
    done
    return 1
}

while :
do
  case "$1" in
    # Install skinny dependencies
    --skinny)
      SKINNY="true"
      shift
      ;;
    # Install ML dependencies
    --ml)
      ML="true"
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Error: unknown option: $1" >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

# Cleanup apt repository to make room for tests.
sudo apt clean
df -h

python --version
pip install --upgrade pip!=25.1 setuptools wheel
pip --version

if [[ "$SKINNY" == "true" ]]; then
  pip install ./libs/skinny
else
  pip install .[extras,gateway,jobs,mcp] --upgrade
fi

req_files=""
# Install Python test dependencies only if we're running Python tests
if [[ "$ML" == "true" ]]; then
  req_files+=" -r requirements/extra-ml-requirements.txt"
fi

if [[ "$SKINNY" == "true" ]]; then
  req_files+=" -r requirements/skinny-test-requirements.txt"
else
  req_files+=" -r requirements/test-requirements.txt"
fi

if [[ ! -z $req_files ]]; then
  retry-with-backoff pip install $req_files
fi

# Install `mlflow-test-plugin`
pip install tests/resources/mlflow-test-plugin

# Print current environment info
pip install aiohttp
which mlflow

# Print mlflow version
mlflow --version

# Turn off trace output & exit-on-errors
set +ex
