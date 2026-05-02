# MLflow Helm Chart

A production-ready Helm chart for deploying [MLflow](https://mlflow.org) on Kubernetes.

## Features

- **MLflow server** with configurable CLI options
- **TLS support** via an existing Kubernetes Secret
- **Persistent storage** with a PersistentVolumeClaim for SQLite or file-based artifact stores
- **Ingress** for external access
- **Prometheus metrics** and optional ServiceMonitor for the Prometheus Operator
- **NetworkPolicy** restricting ingress and egress to required ports
- **RBAC** with independent namespace-scoped (`namespace_rbac`) and cluster-scoped (`cluster_rbac`) rules
- **Garbage collection** via an optional CronJob that runs `mlflow gc`

## Prerequisites

- Kubernetes 1.23+
- Helm 3.8+

## Installation

```bash
helm install mlflow ./charts --namespace mlflow --create-namespace
```

With custom values:

```bash
helm install mlflow ./charts \
  --namespace mlflow \
  --create-namespace \
  -f my-values.yaml
```

## Quick Start: Shared Dev Instance

The simplest way to get a shared MLflow instance running on a cluster — no external database or object store required. MLflow stores metadata in SQLite and artifacts on a PersistentVolumeClaim:

```bash
helm install mlflow ./charts \
  --namespace mlflow \
  --create-namespace \
  --set storage.enabled=true \
  --set mlflow.backendStoreUri="sqlite:////mlflow/mlflow.db" \
  --set mlflow.defaultArtifactRoot="/mlflow/artifacts"
```

Access the UI via port-forward:

```bash
kubectl port-forward -n mlflow svc/mlflow 5000:5000
```

Then open http://localhost:5000 in your browser.

> **Note:** SQLite and local file storage are not suitable for production or high-concurrency use. For production deployments see the [Backend store](#backend-store-metadata-database) and [Artifact store](#artifact-store) sections below.

## Configuration

See [`values.yaml`](./values.yaml) for the full list of configurable parameters.
Common scenarios are described below.

### Backend store (metadata database)

Inline URI (password visible in values — use only for development):

```yaml
mlflow:
  backendStoreUri: "postgresql://user:password@postgres:5432/mlflow"
```

Read from a Kubernetes Secret (recommended for production):

```bash
kubectl create secret generic mlflow-db-secret \
  --from-literal=uri="postgresql://user:password@postgres:5432/mlflow"
```

```yaml
mlflow:
  backendStoreUriFrom:
    secretKeyRef:
      name: mlflow-db-secret
      key: uri
```

### Artifact store

```yaml
mlflow:
  defaultArtifactRoot: "s3://my-bucket/mlflow"
  artifactsDestination: "s3://my-bucket/mlflow"

env:
  - name: AWS_ACCESS_KEY_ID
    valueFrom:
      secretKeyRef:
        name: s3-credentials
        key: access-key-id
  - name: AWS_SECRET_ACCESS_KEY
    valueFrom:
      secretKeyRef:
        name: s3-credentials
        key: secret-access-key
```

### Persistent local storage (SQLite / file store)

```yaml
storage:
  enabled: true
  size: 10Gi

mlflow:
  backendStoreUri: "sqlite:////mlflow/mlflow.db"
  defaultArtifactRoot: "/mlflow/artifacts"
```

### TLS

```bash
kubectl create secret tls mlflow-tls \
  --cert=tls.crt \
  --key=tls.key
```

```yaml
tls:
  enabled: true
  secretName: mlflow-tls
```

### Ingress

MLflow's host-validation middleware only allows `localhost` and private-IP hosts by default.
When exposing MLflow through an Ingress with a public hostname, set `allowed_hosts` to match
that hostname, otherwise requests will be rejected with HTTP 403.

```yaml
server:
  value_options:
    allowed_hosts: "mlflow.example.com"

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: mlflow.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: mlflow-tls
      hosts:
        - mlflow.example.com
```

### Prometheus metrics

```yaml
metrics:
  enabled: true
  path: /metrics

serviceMonitor:
  enabled: true
```

### Garbage collection

Periodically remove soft-deleted runs, experiments, and their artifacts:

```yaml
garbageCollection:
  enabled: true
  schedule: "0 2 * * 0" # weekly at 2 AM on Sunday
  olderThan: "30d" # only remove resources soft-deleted for 30+ days
```

### Resource limits

```yaml
resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 1000m
    memory: 1Gi
```

## Example

See [`example-mlflow-charts.yaml`](./example-mlflow-charts.yaml) for a production-oriented example with a PostgreSQL backend, S3 artifact store, and Secret-based credential injection.

```bash
helm install mlflow ./charts \
  --namespace mlflow \
  --create-namespace \
  -f charts/example-mlflow-charts.yaml
```

## Upgrading

```bash
helm upgrade mlflow ./charts --namespace mlflow -f my-values.yaml
```

## Uninstalling

```bash
helm uninstall mlflow --namespace mlflow
```

> **Note:** `helm uninstall` does not delete PersistentVolumeClaims. If `storage.enabled=true`, delete the PVC manually after uninstalling:
>
> ```bash
> kubectl delete pvc -n mlflow --all
> ```
