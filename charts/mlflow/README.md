# mlflow Helm Chart

MLflow is an open source platform for managing the end-to-end machine 
learning lifecycle.

## TL;DR

```console
helm install mlflow /path/to/chart \
  --set backendStore.existingSecret=mlflow-backend-credentials \
  --set artifacts.s3.defaultArtifactRoot=s3://mlflow \
  --set artifacts.s3.existingSecret=mlflow-artifact-credentials \
  mlflow/mlflow
```

## Introduction

This chart can be used to deploy a mlflow tracking server to a kubernetes
cluster. This chart requires an external artifact storage provider and a
sqlalchemy compatible database; for this reason chart parameters must be
set to configure the artifact storage and database backend.

The helm chart supports the following storage providers:

- AWS S3
- Self hosted MinIO (via S3 configuration parameters)
- Azure File Storage
- Google Cloud Storage

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+

## Installing the Chart

To install the chart with the release name `mlflow`:

```console
helm install mlflow \
  --set backendStore.existingSecret=mlflow-backend-credentials \
  --set artifacts.s3.defaultArtifactRoot=s3://mlflow \
  --set artifacts.s3.existingSecret=mlflow-artifact-credentials \
  mlflow/mlflow
```

> You will likely want to expose mlflow for acces through
> a browser or from your data science environment. This can be configured
> using an ingress or node port. See the `values.yaml` file for more details.

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
  --set artifacts.s3.defaultArtifactRoot=s3://mlflow \
  --set artifacts.s3.existingSecret=mlflow-artifact-credentials \
    mlflow/mlflow
```

Alternatively, a YAML file that specifies the values for the above parameters can be provided while installing the chart. For example,

```console
helm install mlflow -f values.yaml mlflow/mlflow
```
