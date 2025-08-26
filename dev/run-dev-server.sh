#!/bin/bash
set -e

function wait_server_ready {
  for backoff in 0 1 2 4 8; do
    echo "Waiting for tracking server to be ready..."
    sleep $backoff
    if curl --fail --silent --show-error --output /dev/null $1; then
      echo "Server is ready"
      return 0
    fi
  done
  echo -e "\nFailed to launch tracking server"
  return 1
}

mkdir -p outputs
echo 'Running tracking server in the background'

# Handle backend store URI (tracking store)
if [ -n "$MLFLOW_TRACKING_URI" ]; then
  backend_store_uri="--backend-store-uri $MLFLOW_TRACKING_URI"
  default_artifact_root="--default-artifact-root mlruns"
elif [ -n "$MLFLOW_BACKEND_STORE_URI" ]; then
  backend_store_uri="--backend-store-uri $MLFLOW_BACKEND_STORE_URI"
  default_artifact_root="--default-artifact-root mlruns"
else
  backend_store_uri=""
  default_artifact_root=""
fi

# Handle registry store URI (model registry)
if [ -n "$MLFLOW_REGISTRY_URI" ]; then
  registry_store_uri="--registry-store-uri $MLFLOW_REGISTRY_URI"
else
  registry_store_uri=""
fi

if [ ! -d "mlflow/server/js/node_modules" ]; then
  pushd mlflow/server/js
  yarn install
  popd
fi

mlflow server $backend_store_uri $default_artifact_root $registry_store_uri --dev &
wait_server_ready localhost:5000/health
yarn --cwd mlflow/server/js start
