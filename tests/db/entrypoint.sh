#!/bin/bash
set -ex

# Note this script is executed after mounting volumes
if [ -z "$(pip list | grep '^mlflow ')" ]; then
  pip install --no-deps -e .
fi

# For Microsoft SQL server, wait until the database is up and running
if [[ $MLFLOW_TRACKING_URI == mssql* ]]; then
  ./tests/db/init-mssql-db.sh
fi

$@
