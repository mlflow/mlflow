#!/bin/bash

docker build \
    --build-arg MLFLOW_VERSION=${MLFLOW_VERSION:-0.8.2} \
    --build-arg MLFLOW_VERSION_TO_INSTALL=${MLFLOW_VERSION_TO_INSTALL:-0.8.2} \
    -t mlflow_server .

