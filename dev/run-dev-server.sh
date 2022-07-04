#!/bin/bash
set -e

log_file="outputs/mlflow-server.log"

function wait_server_ready {
  for backoff in 0 1 2 4 8; do
    echo "Waiting for server to be ready..."
    sleep $backoff
    if curl --fail --silent --show-error --output /dev/null $1; then
      echo "Server is ready"
      return 0
    fi
  done
  cat $log_file
  echo -e "\nFailed to launch mlflow server"
  return 1
}

mkdir -p outputs
echo 'Running mlflow server in the background'
echo "Logging to $log_file"
if [ -z "$MLFLOW_TRACKING_URI" ]; then
  backend_store_uri=""
  default_artifact_root=""
else
  backend_store_uri="--backend-store-uri $MLFLOW_TRACKING_URI"
  default_artifact_root="--default-artifact-root mlruns"
fi
mlflow server $backend_store_uri $default_artifact_root --gunicorn-opts="--log-level debug" > $log_file 2>&1 &
wait_server_ready localhost:5000/health
yarn --cwd mlflow/server/js start
