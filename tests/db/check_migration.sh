#!/bin/bash
set -ex

cd tests/db

pip install mlflow
python check_migration.py pre-migration
pip install -e ../..
mlflow db upgrade $MLFLOW_TRACKING_URI
python check_migration.py post-migration
