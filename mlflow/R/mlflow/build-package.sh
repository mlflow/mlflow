#!/usr/bin/env bash
set -ex

docker build -f Dockerfile.build -t r-build-package .
docker run --rm --workdir /app -v $(pwd):/app r-build-package Rscript -e 'source(".build-package.R", echo = TRUE)'
