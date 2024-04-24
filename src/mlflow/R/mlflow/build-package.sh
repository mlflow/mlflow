#!/usr/bin/env bash
set -ex

image_name="mlflow-r-dev"

if [ "${USE_R_DEVEL:-false}" = "true" ]
then
  docker_file="Dockerfile.r-devel"
else
  docker_file="Dockerfile.dev"
fi
docker build -f $docker_file -t $image_name .
docker run --rm -v $(pwd):/mlflow/mlflow/R/mlflow $image_name Rscript -e 'source(".build-package.R", echo = TRUE)'
