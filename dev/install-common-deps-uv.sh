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

uv run python --version
uv --version

# Build the list of packages to install
packages=""

# Base packages
packages+=" --upgrade pip!=25.1 setuptools wheel"

# Main package
if [[ "$SKINNY" == "true" ]]; then
  packages+=" ./libs/skinny"
else
  packages+=" .[extras] --upgrade"
fi

# Test requirements
if [[ "$SKINNY" == "true" ]]; then
  packages+=" -r requirements/skinny-test-requirements.txt"
else
  packages+=" -r requirements/test-requirements.txt"
fi

# ML requirements
if [[ "$ML" == "true" ]]; then
  packages+=" -r requirements/extra-ml-requirements.txt"
fi

# Additional packages
packages+=" aiohttp"

# Single uv pip install call for all packages
retry-with-backoff uv pip install $packages

# Install mlflow-test-plugin without dependencies (separate call needed for --no-dependencies)
uv pip install --no-dependencies tests/resources/mlflow-test-plugin

# Print current environment info
uv run which mlflow

# Print mlflow version
uv run mlflow --version

# Turn off trace output & exit-on-errors
set +ex
