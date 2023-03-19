#! /bin/bash

VERSION=2.2.1

docker build \
    -t k3d-mlflow-registry.localhost:12000/mlflow/mlflow:$VERSION \
    --build-arg VERSION=$VERSION \
    --file docker/tracking-server.Dockerfile \
    docker

docker push k3d-mlflow-registry.localhost:12000/mlflow/mlflow:$VERSION
