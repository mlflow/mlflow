#!/usr/bin/env bash

set -ex

pushd ../../mlflow/R/mlflow

image_name="mlflow-r-dev"

docker build -f Dockerfile.dev -t $image_name . --platform linux/amd64

docker run \
  --rm \
  -v $(pwd):/mlflow/mlflow/R/mlflow \
  -v $(pwd)/../../../docs/api_reference/source:/mlflow/docs/api_reference/source \
  $image_name \
  Rscript -e 'source(".build-doc.R", echo = TRUE)'

popd
