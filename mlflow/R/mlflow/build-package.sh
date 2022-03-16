#!/usr/bin/env bash
set -ex

image_name="mlflow-r-dev"

if [ "${USE_DEVEL:-false}" = "true" ]
then
  docker_file="Dockerfile.dev"
else
  docker_file="Dockerfile.devel"
docker build -f $docker_file -t $image_name .
docker run --rm -v $(pwd):/mlflow/mlflow/R/mlflow $image_name Rscript -e 'source(".build-package.R", echo = TRUE)'
