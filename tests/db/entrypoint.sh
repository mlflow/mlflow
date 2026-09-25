#!/bin/bash
set -ex

# Install locked dependencies into a venv outside the mounted repository so the host's `.venv` is
# left alone. Use the image's Python rather than downloading the one pinned in `.python-version`.
export UV_PROJECT_ENVIRONMENT=/opt/venv
export UV_PYTHON=/usr/local/bin/python
# The repository root is mounted to the working directory
uv sync --locked --no-default-groups --extra db --group db-test
uv pip list --python "$UV_PROJECT_ENVIRONMENT"

# For Microsoft SQL server, wait until the database is up and running
if [[ $MLFLOW_TRACKING_URI == mssql* ]]; then
  ./tests/db/init-mssql-db.sh
fi

# Execute the command in the venv
exec uv run --no-sync "$@"
