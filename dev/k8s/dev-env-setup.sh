#!/bin/bash
set -x

REGISTRY_NAME=mlflow-registry.localhost
PORT_RANGE="32000-32002"


# Configure container registry
k3d registry list | grep ^k3d-$REGISTRY_NAME
REGISTRY_EXISTS=$?

if [[ $REGISTRY_EXISTS == 1 ]]; then
    echo "Configuring k3d container image registry 'k3d-$REGISTRY_NAME' for the cluster"
    k3d registry \
        create $REGISTRY_NAME \
        --port 12000
elif [[ $REGISTRY_EXISTS == 0 ]]; then
    echo "The container registry 'k3d-$REGISTRY_NAME' already exists"
fi


# Configure k3d Kubernetes cluster
k3d cluster list | grep ^mlflow
CLUSTER_EXISTS=$?

if [[ $CLUSTER_EXISTS == 1 ]]; then 
    echo "Configuring k3d Kubrnetes cluster 'mlflow'"
    k3d cluster \
        create mlflow \
        --registry-use k3d-$REGISTRY_NAME:12000 \
        --agents 1 \
        -p "$PORT_RANGE:$PORT_RANGE@server:0"
elif [[ $REGISTRY_EXISTS == 0 ]]; then
    echo "The cluster 'mlflow' already exists"
fi
