#!/usr/bin/env bash

# Executes a subset of mlflow tests that is supported with fewer dependencies than the core mlflow package.
# Tests include most client interactions and compatibility points with the mlflow plugins around tracking, projects, models, deployments, and the cli.

# The SQL alchemy store's dependencies are added for a base client/store that can be tested against.
# A different example client/store with a minimal dependency footprint could also work for this purpose.

set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_SKINNY='true'

pytest --verbose tests/test_skinny.py
python -m pip install sqlalchemy alembic sqlparse
pytest --verbose tests/test_runs.py
pytest --verbose tests/tracking/test_client.py
pytest --verbose tests/tracking/test_tracking.py
pytest --verbose tests/projects/test_projects.py
pytest --verbose tests/deployments/test_cli.py
pytest --verbose tests/deployments/test_deployments.py
pytest --verbose tests/projects/test_projects_cli.py

test $err = 0
