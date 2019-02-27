#!/bin/sh
VERSION="0.8.2"
MAINTAINERS="Zak Hassan"
COMPONENT="mlflow-server"

#cleaning up the image folder:

DKR_HUB_NAME=quay.io/zmhassan/mlflow:$VERSION
IMAGE_NAME=mlflow:$VERSION


# To test the docker image run: docker run   -p 5000:5000    quay.io/zmhassan/mlflow:$VERSION
docker build --build-arg mlflow_version=$VERSION --rm -t  $IMAGE_NAME .


docker tag  $IMAGE_NAME $DKR_HUB_NAME
docker push  $DKR_HUB_NAME
