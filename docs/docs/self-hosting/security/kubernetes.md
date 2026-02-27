# Kubernetes Authentication

MLflow includes a built-in request auth provider for Kubernetes environments. This provider automatically adds workspace and authorization headers to outgoing MLflow tracking requests using Kubernetes credentials.

:::important Applicability

This provider is designed for MLflow deployments that use a **custom workspace provider plugin requiring Kubernetes authentication** (e.g., a Kubernetes-based workspace provider included alongside MLflow in a Kubernetes cluster). Different deployments may implement their own Kubernetes workspace providers with varying requirements.

If your MLflow deployment does not use a workspace provider that expects Kubernetes-based identity headers, this provider is not applicable.

:::

## Overview

When enabled, the Kubernetes request auth provider attaches two headers to every MLflow tracking request:

| Header               | Value                                                         |
| -------------------- | ------------------------------------------------------------- |
| `X-MLFLOW-WORKSPACE` | The Kubernetes namespace (from service account or kubeconfig) |
| `Authorization`      | Bearer token (from service account or kubeconfig)             |

These headers allow an MLflow workspace provider plugin to identify the caller's Kubernetes namespace and authenticate the request.

## Prerequisites

Install the `kubernetes` Python package:

```bash
pip install mlflow[kubernetes]
```

## Configuration

Enable the provider by setting the `MLFLOW_TRACKING_AUTH` environment variable:

```bash
export MLFLOW_TRACKING_AUTH=kubernetes
```

No additional configuration is needed. The provider automatically discovers credentials from the environment.

## Credential Sources

The provider tries two credential sources in order, using the first one that succeeds. Both the namespace and token are always sourced from the same location to ensure consistency.

### 1. Service Account (in-cluster)

When running inside a Kubernetes pod, the provider reads the mounted service account files:

- **Namespace**: `/var/run/secrets/kubernetes.io/serviceaccount/namespace`
- **Token**: `/var/run/secrets/kubernetes.io/serviceaccount/token`

This is the default path for pods with a service account mounted (which is the standard Kubernetes behavior).

### 2. Kubeconfig

When service account files are not available, the provider falls back to kubeconfig:

- **Namespace**: Extracted from the active context
- **Token**: Resolved via the Kubernetes Python client's `ApiClient`, which handles exec-based auth flows (EKS, GKE, AKS, OpenShift, OIDC)

## Usage

Once configured, no code changes are needed. All MLflow tracking calls will automatically include the authorization headers with values sourced from Kubernetes:

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```

If a workspace is explicitly set (e.g., via `mlflow.set_workspace()`), it takes priority over the Kubernetes namespace. The namespace from Kubernetes credentials is used as a default when no workspace is specified.
