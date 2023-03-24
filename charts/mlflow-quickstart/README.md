# mlflow Quickstart Helm Chart

MLflow is an open source platform for managing the end-to-end machine learning lifecycle.

## Introduction

This chart quickly deploys a mlflow tracking server instance, MinIO (artifact storage) 
and PostgreSQL (database persistance). If you would like to use another database or
artifact storage provider, consider using the mlflow chart.

This chart has three dependencies:

- mlflow chart (this repository)
- MinIO chart (MinIO repository)
- PostgreSQL chart (bitnami repository).

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- PV provisioner support in the underlying infrastructure

## Installing the Chart

To install the chart with the release name `mlflow`:

```bash
helm install mlflow .
```

> Depending on your cluster, you will likely want to set the following values:
>
> - `minio.persistence.storageClass`
> - `mlflow.service.type=NodePort` or configure ingress
> - `minio.service.type-NodePort` or configure ingress

The command deploys mlflow, minio and postgresql on the Kubernetes
cluster in the default configuration.

> :warning: If you want to use this in a production setting, please change
> MinIO and PostgreSQL default credentials.

## Uninstalling the Chart

To uninstall/delete the `mlflow` release:

```bash
helm delete mlflow
```

The command removes all the Kubernetes components associated with the chart and deletes the release.

## Parameters

See `values.yaml` for all the helm chart parameters and descriptions

Specify each parameter using the `--set key=value[,key=value]` argument to `helm install`. For example,

```bash
helm install mlflow . \
  --set backendStore.existingSecret=mlflow-backend-credentials
```

Alternatively, a YAML file that specifies the values for the above parameters can be provided while installing the chart. For example,

```bash
helm install mlflow . --values values.yaml
```
