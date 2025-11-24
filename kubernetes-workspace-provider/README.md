# MLflow Kubernetes Workspace Provider

This package ships two MLflow extensions backed by Kubernetes: a workspace provider that lists
namespaces as MLflow workspaces and an optional authorization plugin that enforces Kubernetes RBAC
for every MLflow API request. Each MLflow workspace maps 1:1 to a Kubernetes namespace.

## Features

### Kubernetes workspace provider

- Lists Kubernetes namespaces as MLflow workspaces and keeps them cached via a background watch
  loop.
- Hides Kubernetes system namespaces (`kube-*`, `openshift-*`) and lets you add extra glob filters.
- Supports an optional label selector so that only namespaces marked for MLflow are exposed.
- Populates workspace descriptions from the `mlflow.kubeflow.org/workspace-description` annotation
  when present.
- Automatically loads Kubernetes credentials from the in-cluster service account or a local
  `~/.kube/config`.
- Provides a read-only experience: namespace lifecycle is owned outside of MLflow and all CRUD calls
  raise `NotImplementedError`.

### Kubernetes authorization plugin (`kubernetes-auth`)

- Accepts Kubernetes service-account (or workload identity) tokens via the standard
  `Authorization: Bearer <token>` header or the `X-Forwarded-Access-Token` header when running
  behind a proxy.
- Evaluates Kubernetes `SelfSubjectAccessReview` objects for each MLflow API call across the
  `experiments`, `registeredmodels`, `workspaces`, and `jobs` resources in the `mlflow.kubeflow.org`
  API group.
- Transparently rewrites run requests so the authenticated user becomes the run owner.
- Filters workspace listings to the set of namespaces the caller can `list`.
- Denies workspace create/update/delete operations, even if RBAC would otherwise allow them, keeping
  the namespace lifecycle external to MLflow.
- Covers both Flask and FastAPI routes exposed by the MLflow server and caches authorization
  decisions for a configurable TTL.

## Components

| Entry point       | MLflow hook                 | Description                                                                                                            |
| ----------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `kubernetes`      | `mlflow.workspace_provider` | Instantiates `KubernetesWorkspaceProvider`, which maps MLflow workspaces to Kubernetes namespaces.                     |
| `kubernetes-auth` | `mlflow.app`                | Wraps the MLflow tracking server with authorization middleware that consults Kubernetes RBAC before serving a request. |

## Prerequisites

- MLflow 3.6+ with the workspaces feature flag enabled (via `--enable-workspaces` or
  `MLFLOW_ENABLE_WORKSPACES=1`).
- A Kubernetes cluster reachable from the MLflow tracking server.
- The service account used by the MLflow server must be allowed to list and watch namespaces.
- Users (or service accounts acting on their behalf) need permissions in the `mlflow.kubeflow.org`
  API group that align with the operations they perform (see
  [Kubernetes RBAC requirements](#kubernetes-rbac-requirements)).
- `kubectl` 1.24+ if you rely on `kubectl create token` to mint service-account tokens. Older
  clusters may require manually creating token Secrets.

## Installation

```bash
pip install ./kubernetes-workspace-provider
```

For local development:

```bash
pip install -e "./kubernetes-workspace-provider[dev]"
```

## Quick start

### 1. Install MLflow and the provider

Ensure the MLflow server environment has both packages available:

```bash
pip install mlflow
pip install ./kubernetes-workspace-provider
```

### 2. Prepare Kubernetes namespaces and RBAC

Optionally, label the namespaces you want to expose (e.g., `mlflow-enabled=true`) and, optionally,
add the `mlflow.kubeflow.org/workspace-description` annotation for human-friendly descriptions.
Grant the MLflow server service account permissions as shown in the
[example manifest](#example-manifest).

### 3. Configure the server

Either set environment variables (e.g. `MLFLOW_K8S_WORKSPACE_LABEL_SELECTOR="mlflow-enabled=true"`)
or pass query parameters on the workspace store URI. The examples below use environment variables so
the server flags stay short.

### 4. Start MLflow with the provider and plugin

```bash
# Replace these with your own selector and default namespace.
export MLFLOW_K8S_WORKSPACE_LABEL_SELECTOR="mlflow-enabled=true"
export MLFLOW_K8S_DEFAULT_WORKSPACE="team-a"

mlflow server \
  --backend-store-uri sqlite:///$(pwd)/mlflow.db \
  --default-artifact-root ./mlruns-artifacts \
  --enable-workspaces \
  --workspace-store-uri "kubernetes://" \
  --app-name kubernetes-auth
```

SQLite keeps the quick start self-contained; switch to a managed database and durable artifact store
for production deployments. Omit `--app-name kubernetes-auth` if you do not need the Kubernetes RBAC
plugin.

### 5. Call the API with a Kubernetes token

```bash
TOKEN=$(kubectl -n team-a create token mlflow-writer)
curl \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "X-MLFLOW-WORKSPACE: team-a" \
  http://<mlflow-host>/api/2.0/mlflow/experiments/search
```

When running behind a reverse proxy, forward the token via `X-Forwarded-Access-Token` instead of (or
in addition to) the `Authorization` header.

## Configuration

`KubernetesWorkspaceProvider` reads its options from (1) environment variables, (2) keyword
arguments passed to the provider constructor, and (3) query parameters on the `kubernetes://`
workspace store URI. Later sources override earlier ones, so a query parameter always wins over an
environment variable.

### Workspace provider options

| Variable                              | Default | Description                                                                                                                                                  |
| ------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `MLFLOW_K8S_WORKSPACE_LABEL_SELECTOR` | empty   | Optional Kubernetes label selector applied when listing namespaces (for example `mlflow-enabled=true,environment in (prod,staging)`).                        |
| `MLFLOW_K8S_DEFAULT_WORKSPACE`        | empty   | Optional workspace (namespace) name used when a request omits `X-MLFLOW-WORKSPACE`. When unset, clients must always send the header or call `set_workspace`. |
| `MLFLOW_K8S_NAMESPACE_EXCLUDE_GLOBS`  | empty   | Extra comma-separated glob patterns (e.g. `team-secret-*,*-internal`) to hide alongside the built-in `kube-*` and `openshift-*` exclusions.                  |

Label selectors use the standard Kubernetes string format, so multiple key/value clauses are simply
comma-separated. Namespaces that match the built-in or custom exclude globs are hidden from both
listings and lookups—referencing them later will raise a "workspace not found" error. If
`MLFLOW_K8S_DEFAULT_WORKSPACE` is not set and a request omits a workspace header or CLI override,
the MLflow server returns an `INVALID_PARAMETER_VALUE` error rather than guessing.

### Authorization plugin options

| Variable                                  | Default                      | Description                                                                                                                                                             |
| ----------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MLFLOW_K8S_AUTH_CACHE_TTL_SECONDS`       | `300`                        | TTL (in seconds) for cached access-review decisions. Must be a positive number if set.                                                                                  |
| `MLFLOW_K8S_AUTH_USERNAME_CLAIM`          | `sub`                        | JWT/OIDC claim to treat as the authenticated username. This value is injected into run payloads as `user_id`.                                                           |
| `MLFLOW_K8S_AUTH_AUTHORIZATION_MODE`      | `self_subject_access_review` | Determines whether the plugin performs `SelfSubjectAccessReview` (default) using the caller's token or `SubjectAccessReview` using headers provided by a trusted proxy. |
| `MLFLOW_K8S_AUTH_REMOTE_USER_HEADER`      | `x-remote-user`              | HTTP header (case-insensitive) that carries the authenticated username when `subject_access_review` mode is enabled (aligned with kube-rbac-proxy defaults).            |
| `MLFLOW_K8S_AUTH_REMOTE_GROUPS_HEADER`    | `x-remote-groups`            | HTTP header that lists the user's groups when `subject_access_review` mode is enabled.                                                                                  |
| `MLFLOW_K8S_AUTH_REMOTE_GROUPS_SEPARATOR` | `\|`                         | Separator used to split the groups header into individual entries (matches kube-rbac-proxy's `--auth-header-groups-field-separator`).                                   |

#### Using kube-rbac-proxy headers

Reverse proxies such as [`kube-rbac-proxy`](https://github.com/brancz/kube-rbac-proxy) can
authenticate end users but do not always forward their bearer tokens. Set
`MLFLOW_K8S_AUTH_AUTHORIZATION_MODE=subject_access_review` so the MLflow plugin trusts the proxy's
headers and performs a Kubernetes `SubjectAccessReview` on behalf of the reported user:

```bash
export MLFLOW_K8S_AUTH_AUTHORIZATION_MODE=subject_access_review
# Override these if kube-rbac-proxy is configured with custom names/separators
export MLFLOW_K8S_AUTH_REMOTE_USER_HEADER="x-remote-user"
export MLFLOW_K8S_AUTH_REMOTE_GROUPS_HEADER="x-remote-groups"
export MLFLOW_K8S_AUTH_REMOTE_GROUPS_SEPARATOR="|"
```

In this mode the MLflow server's own service account contacts the Kubernetes API, so it must be
allowed to `create` the `subjectaccessreviews.authorization.k8s.io` resource in addition to the
namespace watch permissions described below. The user and group information from the headers becomes
the authoritative identity (and is also injected into run payloads), so be sure only a trusted proxy
can reach MLflow.

### Workspace store URI parameters

Pass these as query parameters to the `kubernetes://` workspace store URI. URL-encode values as
needed.

- `label_selector`: overrides `MLFLOW_K8S_WORKSPACE_LABEL_SELECTOR`.
- `default_workspace`: overrides `MLFLOW_K8S_DEFAULT_WORKSPACE`.
- `namespace_exclude_globs`: comma-separated patterns that augment the built-in exclusions.

Leave a parameter empty (e.g., `label_selector=`) to defer to environment variables.

## Running the MLflow server

`KubernetesWorkspaceProvider` automatically tries `config.load_incluster_config()` and falls back to
`config.load_kube_config()`, so the MLflow server can run either inside a cluster (using its service
account token) or from a workstation that has a valid `~/.kube/config`.

Enable workspaces and reference the provider by scheme:

```bash
mlflow server \
  --backend-store-uri postgresql://... \
  --default-artifact-root s3://mlflow-artifacts \
  --enable-workspaces \
  --workspace-store-uri "kubernetes://?label_selector=mlflow-enabled%3Dtrue&namespace_exclude_globs=team-secret-*,*-internal"
```

Add `--app-name kubernetes-auth` to activate the authorization plugin. When MLflow runs under
`uvicorn`, the plugin automatically injects FastAPI middleware so OTLP trace ingestion and Job APIs
are also protected.

## Sending requests

Clients must supply **both** a workspace context and a bearer token.

### Workspace context options

- Set the `X-MLFLOW-WORKSPACE` header on every HTTP request.
- Call `mlflow.set_workspace("team-a")` in the Python client.
- Export `MLFLOW_WORKSPACE=team-a` before using the CLI.

### Authentication options

- Provide `Authorization: Bearer <token>` (or `X-Forwarded-Access-Token`) explicitly when calling
  the REST API.
- Export `MLFLOW_TRACKING_TOKEN=$(kubectl -n team-a create token mlflow-writer)` so MLflow CLIs and
  SDKs automatically send the bearer token without additional flags.

If neither a workspace header nor `MLFLOW_K8S_DEFAULT_WORKSPACE` is provided, the server fails the
request with `INVALID_PARAMETER_VALUE` before executing any work.

Example request:

```bash
TOKEN=$(kubectl -n team-a create token mlflow-experiments-reader)
curl \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "X-MLFLOW-WORKSPACE: team-a" \
  http://<mlflow-host>/api/2.0/mlflow/runs/search
```

Reverse proxies such as Istio or OAuth2 proxies can forward the caller token via
`X-Forwarded-Access-Token`, which the plugin treats the same as `Authorization`.

## Development

In a fresh virtual environment, align tool versions and generate protobuf stubs before installing
the editable package:

```bash
uv self update 0.9.8
bash dev/generate-protos.sh
pip install -e "./kubernetes-workspace-provider[dev]"
pytest kubernetes-workspace-provider/tests
```

`uv self update 0.9.8` matches the version MLflow currently pins for docs builds; skipping it can
lead to dependency resolution differences.

## Kubernetes RBAC requirements

Both the workspace provider and the authorization plugin communicate with the Kubernetes control
plane. Grant the MLflow server service account permission to list and watch namespaces:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: mlflow-k8s-workspace-provider
rules:
  - apiGroups: [""]
    resources: ["namespaces"]
    verbs: ["list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: mlflow-k8s-workspace-provider
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: mlflow-k8s-workspace-provider
subjects:
  - kind: ServiceAccount
    name: <service-account-name>
    namespace: <service-account-namespace>
```

The authorization plugin evaluates Kubernetes `SelfSubjectAccessReview` requests against the
`mlflow.kubeflow.org` API group. Tokens presented to the MLflow API must be authorized for the
following for full access:

| Resource           | Verbs                                                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `experiments`      | `get`, `list`, `create`, `update`, `delete`                                                                            |
| `registeredmodels` | `get`, `list`, `create`, `update`, `delete`                                                                            |
| `workspaces`       | `get`, `list`, `create`, `update`, `delete` (mutations are denied by the plugin but RBAC should still describe intent) |
| `jobs`             | `get`, `list`, `create`                                                                                                |

> **Note:** Prompts share storage and permissions with registered models. Granting access to the
> `registeredmodels` resource automatically covers prompt operations; no separate `prompts` RBAC
> entry is required.

For workloads restricted to a single namespace, grant the service account these permissions with a
`Role` and `RoleBinding`:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mlflow-k8s-namespace-access
  namespace: <workspace-namespace>
rules:
  - apiGroups: ["mlflow.kubeflow.org"]
    resources:
      - experiments
      - registeredmodels
      - workspaces
      - jobs
    verbs:
      - get
      - list
      - create
      - update
      - delete
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mlflow-k8s-namespace-access
  namespace: <workspace-namespace>
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: mlflow-k8s-namespace-access
subjects:
  - kind: ServiceAccount
    name: <service-account-name>
    namespace: <service-account-namespace>
```

When `MLFLOW_K8S_AUTH_AUTHORIZATION_MODE=subject_access_review`, the MLflow server itself performs
Kubernetes `SubjectAccessReview` calls using its service account. Grant that account permission to
`create` the `subjectaccessreviews.authorization.k8s.io` resource (cluster-scoped or namespaced via
an aggregated `Role`) in addition to the namespace list/watch permissions shown earlier.

### Example manifest

The following manifest provisions a `team-a` namespace annotated for MLflow, service accounts for
writers and read-only experiment access, and the necessary RBAC bindings:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: team-a
  annotations:
    mlflow.kubeflow.org/workspace-description: "Workspace for my team"
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mlflow-writer
  namespace: team-a
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mlflow-experiments-reader
  namespace: team-a
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mlflow-k8s-namespace-access
  namespace: team-a
rules:
  - apiGroups:
      - mlflow.kubeflow.org
    resources:
      - experiments
      - registeredmodels
      - workspaces
      - jobs
    verbs:
      - get
      - list
      - create
      - update
      - delete
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mlflow-k8s-namespace-access
  namespace: team-a
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: mlflow-k8s-namespace-access
subjects:
  - kind: ServiceAccount
    name: mlflow-writer
    namespace: team-a
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mlflow-experiments-readonly
  namespace: team-a
rules:
  - apiGroups:
      - mlflow.kubeflow.org
    resources:
      - experiments
      - jobs
    verbs:
      - get
      - list
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mlflow-experiments-readonly
  namespace: team-a
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: mlflow-experiments-readonly
subjects:
  - kind: ServiceAccount
    name: mlflow-experiments-reader
    namespace: team-a
```

Apply the manifest with:

```bash
kubectl apply -f team-a-workspace.yaml
```

### Generating tokens for service accounts

To obtain an authentication token for either service account (requires Kubernetes 1.24+):

```bash
kubectl -n team-a create token mlflow-writer
kubectl -n team-a create token mlflow-experiments-reader
```

Include the resulting token in the `Authorization: Bearer <token>` header (or
`X-Forwarded-Access-Token` when a proxy handles authentication) along with the `X-MLFLOW-WORKSPACE`
header for every MLflow request. Older clusters may not support `kubectl create token`. In that
case, consult the Kubernetes documentation on manually provisioning service-account token Secrets.
