#!/bin/bash
set -ex

cd tests/db

# Install the lastest version of mlflow from PyPI
uv pip install --system mlflow
python check_migration.py pre-migration
# Install mlflow from the repository
uv pip install --system -e ../..
mlflow db upgrade $MLFLOW_TRACKING_URI
python check_migration.py post-migration
