#!/usr/bin/env bash

# This script is a uv-based version of dev/install-common-deps.sh, replacing pip/pipenv with uv for faster dependency management.

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

# For skinny mode, set environment variable to avoid discovering root pyproject.toml
# This needs to be set early so all uv commands respect it
if [[ "$SKINNY" == "true" ]]; then
  export UV_NO_CONFIG=1
fi

# Cleanup apt repository to make room for tests.
sudo apt clean
df -h

# Ensure virtual environment exists (uv will reuse if already present)
uv venv

# Activate the virtual environment for non-uv commands
source .venv/bin/activate

# Don't use 'uv run' here as it would install the root project dependencies
python --version
uv --version

# Build the list of packages to install
packages=""

# Base packages
packages+="pip!=25.1 setuptools wheel"

# Main package
if [[ "$SKINNY" == "true" ]]; then
  packages+=" ./libs/skinny"
else
  packages+=" .[extras]"
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

# Add virtualenv for model isolation (needed by MLflow for creating model environments)
packages+=" virtualenv"

# Single uv pip install call for all packages with constraints
uv pip install -c requirements/constraints.txt --upgrade $packages

# Install mlflow-test-plugin without dependencies (separate call needed for --no-deps)
uv pip install -c requirements/constraints.txt --no-deps tests/resources/mlflow-test-plugin

# Print current environment info
which mlflow

# Print mlflow version  
mlflow --version

# Turn off trace output & exit-on-errors
set +ex
