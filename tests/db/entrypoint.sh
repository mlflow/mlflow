#!/bin/bash
set -ex

# Install mlflow (assuming the repository root is mounted to the working directory)
if [ "$INSTALL_MLFLOW_FROM_REPO" = "true" ]; then
  pip install --no-deps -e .
fi

# For Microsoft SQL server, wait until the database is up and running
if [ "$INSTALL_ODBC_DRIVER_18" = "true" ]; then
  ./tests/db/init-mssql18-db.sh
elif [[ $MLFLOW_TRACKING_URI == mssql* ]]; then
  ./tests/db/init-mssql-db.sh
fi

# Execute the command
exec "$@"
