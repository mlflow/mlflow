#!/bin/bash
set -ex

cd tests/db

# Run the pre-migration step with the latest mlflow from PyPI in an isolated environment, pinning
# its dependencies to the locked versions
locked=$(mktemp)
uv export --locked --no-default-groups --extra db --group db-test \
  --no-emit-workspace --no-hashes --output-file "$locked"
uv run --isolated --no-project --with mlflow --with-requirements "$locked" \
  python check_migration.py pre-migration
# Run the post-migration step with mlflow from the repository
uv run --no-sync mlflow db upgrade $MLFLOW_TRACKING_URI
uv run --no-sync python check_migration.py post-migration
