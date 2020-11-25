#!/usr/bin/env bash
set -x
set -e

mlflow db upgrade "$JOBSERVER_RDB_URL"

mlflow server \
    --backend-store-uri "$JOBSERVER_RDB_URL" \
    --default-artifact-root "s3://grnds-$AWS_ENVIRONMENT-cortex-shared/mlflow" \
    --host 0.0.0.0 \
    --port 5000

