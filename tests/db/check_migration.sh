#!/bin/bash
set -ex

cd tests/db

# Install the lastest version of mlflow from PyPI
# pip install mlflow
# Install mlflow from the repository
pip install -e ../..
python check_migration.py pre-migration
mlflow db upgrade $MLFLOW_TRACKING_URI
python check_migration.py post-migration
