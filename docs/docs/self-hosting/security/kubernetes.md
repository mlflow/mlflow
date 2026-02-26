# Kubernetes Authentication

MLflow includes a built-in request auth provider for Kubernetes environments. This provider automatically adds workspace and authorization headers to outgoing MLflow tracking requests using Kubernetes credentials.

:::important Applicability

This provider is designed for MLflow deployments that use a **custom workspace provider plugin requiring Kubernetes authentication** (e.g., a Kubernetes-based workspace provider included alongside MLflow in a Kubernetes cluster). Different deployments may implement their own Kubernetes workspace providers with varying requirements.

If your MLflow deployment does not use a workspace provider that expects Kubernetes-based identity headers, this provider is not applicable.

:::

## Overview

When enabled, the Kubernetes request auth provider attaches two headers to every MLflow tracking request:

| Header               | Value                                                         |
|----------------------|---------------------------------------------------------------|
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

### 2. Kubeconfig (local development)

When service account files are not available (e.g., running locally), the provider falls back to kubeconfig:

- **Namespace**: Extracted from the active context
- **Token**: Resolved via the Kubernetes Python client's `ApiClient`, which handles exec-based auth flows (EKS, GKE, AKS, OpenShift, OIDC)

This allows developers to test against a remote MLflow instance using their local Kubernetes credentials.

## Caching

File reads for service account credentials are cached with a 1-minute TTL to avoid repeated I/O on every tracking request.

## Usage

Once configured, no code changes are needed. All MLflow tracking calls automatically include the Kubernetes authentication headers:

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```

### Header Behavior

- If both `X-MLFLOW-WORKSPACE` and `Authorization` headers are already present on a request, the provider skips credential lookup entirely.
- If only one header is present, the provider fills in the missing one from Kubernetes credentials.
- If credentials cannot be determined from either source, an `MlflowException` is raised.

## Creating a Custom Request Auth Provider

If the built-in Kubernetes provider does not fit your needs, you can create your own request auth provider plugin. See the [Plugins documentation](/classic-ml/plugins#authentication-plugins) for details on implementing and registering a custom `RequestAuthProvider`.
