#!/bin/bash
set -ex

# Install mlflow
pip install --no-deps -e .

# For Microsoft SQL server, wait until the database is up and running
if [[ $MLFLOW_TRACKING_URI == mssql* ]]; then
  ./tests/db/init-mssql-db.sh
fi

# Run the command
"$@"
