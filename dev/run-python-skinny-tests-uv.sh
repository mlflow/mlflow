#!/usr/bin/env bash

# Executes a subset of mlflow tests that is supported with fewer dependencies than the core mlflow package.
# Tests include most client interactions and compatibility points with the mlflow plugins around tracking, projects, models, deployments, and the cli.
# 
# This script is a uv-based version of dev/run-python-skinny-tests.sh, replacing pip/pipenv with uv for faster dependency management.

# The SQL alchemy store's dependencies are added for a base client/store that can be tested against.
# A different example client/store with a minimal dependency footprint could also work for this purpose.

set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_SKINNY='true'

# Set UV_NO_CONFIG to avoid discovering root pyproject.toml
# This prevents uv from installing the main mlflow dependencies
export UV_NO_CONFIG=1

uv run --no-project pytest tests/test_skinny_client_omits_sql_libs.py

# After verifying skinny client does not include store specific requirements,
# we are installing sqlalchemy store requirements as our example store for the test suite.
# SQL Alchemy serves as a simple, fully featured option to test skinny client store scenarios.
uv pip install sqlalchemy alembic cryptography

# Given the example store does not delete dependencies, we verify non store related dependencies
# after the example store setup. This verifies both the example store and skinny client do not add
# unintended libraries.
uv run --no-project pytest tests/test_skinny_client_omits_data_science_libs.py

# Install numpy that is required by mlflow.types.schema and pre-installed in DBR.
uv pip install numpy

uv run --no-project pytest \
  tests/test_runs.py \
  tests/tracking/test_client.py \
  tests/tracking/test_tracking.py \
  tests/projects/test_projects.py \
  tests/deployments/test_cli.py \
  tests/deployments/test_deployments.py \
  tests/projects/test_projects_cli.py \
  tests/utils/test_requirements_utils.py::test_infer_requirements_excludes_mlflow \
  tests/utils/test_search_utils.py \
  tests/store/tracking/test_file_store.py \
  tests/utils/test_doctor.py \
  --import-mode=importlib

uv pip install pandas
uv run --no-project pytest tests/test_skinny_client_autolog_without_scipy.py

test $err = 0
