#!/usr/bin/env bash
set -ex

image_name="mlflow-r-dev"
docker build -f Dockerfile.dev -t $image_name .
docker run --rm -v $(pwd):/mlflow $image_name Rscript -e 'source(".build-package.R", echo = TRUE)'
