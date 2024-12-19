#!/usr/bin/env bash

set -ex

pushd ../../mlflow/R/mlflow

image_name="mlflow-r-dev"

# Workaround for this issue:
# https://discuss.circleci.com/t/increased-rate-of-errors-when-pulling-docker-images-on-machine-executor/42094
n=0
until [ "$n" -ge 3 ]
do
  docker build -f Dockerfile.dev -t $image_name . --platform linux/amd64 && break
  n=$((n+1))
  sleep 5
done

docker run \
  --rm \
  -v $(pwd):/mlflow/mlflow/R/mlflow \
  -v $(pwd)/../../../docs/api_reference/source:/mlflow/docs/api_reference/source \
  $image_name \
  Rscript -e 'source(".build-doc.R", echo = TRUE)'

popd
