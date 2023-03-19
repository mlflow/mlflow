#! /bin/bash

mkdir ~/.kube


k3d registry \
    create mlflow-registry.localhost \
    --port 12000


k3d cluster \
    create mlflow \
    --registry-use k3d-mlflow-registry.localhost:12000 \
    --agents 1 \
    -p "32000-32002:32000-32002@server:0"


bash dev/kubernetes/build-image.sh
