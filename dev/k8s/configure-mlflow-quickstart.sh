#!/bin/bash
set -ex

helm repo add minio https://charts.min.io/
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

helm -n mlflow \
    upgrade \
    --install \
    --create-namespace \
    mlflow-quickstart \
    charts/mlflow-quickstart \
    --set minio.persistence.storageClass=local-path \
    --set mlflow.image.repository=k3d-mlflow-registry.localhost:12000/mlflow/mlflow \
    --set mlflow.image.pullPolicy=Always \
    --set mlflow.service.type=NodePort \
    --set minio.service.type=NodePort
