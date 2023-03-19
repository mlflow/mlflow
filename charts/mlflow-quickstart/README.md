# mlflow Quickstart Helm Chart

MLflow is an open source platform for managing the end-to-end machine learning lifecycle.

## Introduction

This chart quickly deploys a mlflow tracking server instance along
with MinIO (artifact storage) and PostgreSQL (database persistance)
using sub charts. If you would like to use another database or
artifact storage provider use the mlflow chart.

This chart has three dependencies:

- The mlflow chart (this repository)
- A MinIO chart (MinIO repository)
- A PostgreSQL chart (bitnami).

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- PV provisioner support in the underlying infrastructure

## Installing the Chart

To install the chart with the release name `mlflow`:

```console
helm install mlflow /path/to/chart
```
> Depending on your cluster, you will likely want to set the following values:
> - `minio.persistence.storageClass`
> - `mlflow.service.type=NodePort` or configure ingress
> - `minio.service.type-NodePort` or configure ingress

The command deploys mlflow, minio and postgresql on the Kubernetes
cluster in the default configuration.

> :warning: If you want to use this in a production setting, please change
> MinIO and PostgreSQL default credentials.

## Uninstalling the Chart

To uninstall/delete the `mlflow` deployment:

```console
helm delete mlflow
```

The command removes all the Kubernetes components associated with the chart and deletes the release.

## Parameters

See `values.yaml` for all the helm chart parameters and descriptions

Specify each parameter using the `--set key=value[,key=value]` argument to `helm install`. For example,

```console
helm install mlflow \
  --set backendStore.existingSecret=mlflow-backend-credentials \
  mlflow-quickstart/mlflow
```

Alternatively, a YAML file that specifies the values for the above parameters can be provided while installing the chart. For example,

```console
helm install mlflow -f values.yaml mlflow/mlflow
```
