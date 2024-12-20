# mlflow

![mlflow](https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/_static/MLflow-logo-final-black.png)

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 2.14.1](https://img.shields.io/badge/AppVersion-2.14.1-informational?style=flat-square)

A Helm chart for MLflow - Open source platform for the machine learning lifecycle.

**Homepage:** <https://github.com/mlflow/mlflow/tree/master/charts/mlflow>

## Prerequisites

- Kubernetes >= 1.19
- Helm >= 3.2.0

## Usage

### Add Helm Repo

```bash
helm repo add mlflow https://mlflow.github.io/mlflow
helm repo update
```

See [helm repo](https://helm.sh/docs/helm/helm_repo) for command documentation.

### Install Chart

```bash
helm install [RELEASE_NAME] mlflow/mlflow [flags]
```

See [helm install](https://helm.sh/docs/helm/helm_install) for command documentation.

See [configurations](#configurations) below.

See [values](#values) below.

### Upgrade Chart

```bash
helm upgrade [RELEASE_NAME] mlflow/mlflow [flags]
```

See [helm upgrade](https://helm.sh/docs/helm/helm_upgrade) for command documentation.

### Uninstall Chart

```bash
helm uninstall [RELEASE_NAME]
```

This removes all the Kubernetes resources associated with the chart and deletes the release.

See [helm uninstall](https://helm.sh/docs/helm/helm_uninstall) for command documentation.

## Configurations

### Tracking Server

MLflow tracking server is a stand-alone HTTP server that serves multiple REST API endpoints for tracking runs/experiments. For more information, please visit [MLflow Tracking Server](https://mlflow.org/docs/latest/tracking/server.html).

#### Using the Tracking Server for Proxied Artifact Access

By default, the tracking server stores artifacts in its local filesystem under `./mlartifacts` directory. To configure the tracking server to connect to remote storage and serve artifacts:

```yaml
trackingServer:
  mode: serve-artifacts
  artifactsDestination: s3://my-bucket
```

With this setting, MLflow server works as a proxy for accessing remote artifacts. The MLflow clients make HTTP request to the server for fetching artifacts.

#### Using the Tracking Server without Proxying artifact Access

If you want to directly access remote storage without proxying through the tracking server. In this case, you can set `trackingServer.mode` to `no-serve-artifacts` as follows:

```yaml
trackingServer:
  mode: no-serve-artifacts
  defaultArtifactRoot: s3://my-bucket
```

#### Using the Tracking Server For Artifacts Only

If the volume of tracking server requests is sufficiently large and performance issues are noticed, a tracking server can be configured to serve in `artifacts-only` mode. This configuration ensures that the processing of artifacts is isolated from all other tracking server event handling:

```yaml
trackingServer:
  mode: artifacts-only
  artifactsDestination: s3://my-bucket
```

#### Configure Default Artifact Root

The default artifact root is a directory in which to store artifacts for any new experiments created. For backend store that uses a database, this option is required.

The following configuration enables artifact store and set default artifact root as `./mlruns`:

```yaml
artifactStore:
  defaultArtifactRoot: ./mlruns
```

### Backend Store

The backend store is a core component where MLflow stores metadata for runs and experiments. By default, MLflow stores metadata in local `./mlruns` directory. Supported backend store types are:

- Local file path e.g. `file:/my/local/dir`
- A database, mlflow now supports `mysql`, `mssql`, `sqlite`, `postgresql`
- HTTP server e.g. `https://my-server:5000`
- Databricks workspace

:warning: **Important**:

- In order to use [Model Registry](https://mlflow.org/docs/latest/model-registry.html#registry) functionality, you must use a database as backend store.
- `mlflow server` command will fail against a database backend store with an out-of-date database schema. To prevent this, you must set `backendStore.databaseUpgrade` to `true` and an init container will be added to execute `mlflow db upgrade ${MLFLOW_BACKEND_STORE_URI}`. Schema migrations can result in database downtime, may take longer on larger databases, and are not guaranteed to be transactional. You should always take a backup of your database prior to running mlflow db upgrade.

For more information, please visit [MLflow Backend Stores](https://mlflow.org/docs/latest/tracking/backend-stores.html).

#### Configure Backend Store URI

##### Option #1: Read Backend Store URI From an Existing Secret

The existing secret should contain key `MLFLOW_BACKEND_STORE_URI`. If you do not already have one, you can create a new secret as follows:

```bash
kubectl create secret generic mlflow-backend-store-secret \
    --from-literal=MLFLOW_BACKEND_STORE_URI=<YOUR_BACKEND_STORE_URI>
```

Then configure `backendStore.existingSecret` to use the secret just created:

```yaml
backendStore:
  existingSecret: mlflow-backend-store-secret
```

##### Option #2: Configure Backend Store URI Directly

If no existing secret is specified, `backendStore.createSecret.backendStoreUri` can be configured directly to specify backend store URI and a new secret will be created to hold it:

```yaml
backendStore:
  createSecret:
    backendStoreUri: <YOUR_MLFLOW_BACKEND_STORE_URI>
```

#### Use Database as Backend Store

If you want to use a database as backend store, then the backend store URI should be encoded as `<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>`. For more details, please visit [SQLAlchemy Database URL](https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls).

### Artifact Store

The artifact store is a core component where MLflow stores artifacts for each run such as model weights, images, model and data files. Supported storage types for the artifact store are:

- Amazon S3 and S3-compatible storage e.g. `s3://<bucket>/<path>`
- Azure Blob Storage e.g. `wasbs://<container>@<storage-account>.blob.core.windows.net/<path>`
- Google Cloud Storage e.g. `gs://<bucket>/<path>`
- Alibaba Cloud OSS e.g. `oss://<bucket>/<path>`
- FTP server e.g. `ftp://user:pass@host/path/to/directory`
- SFTP Server e.g. `sftp://user@host/path/to/directory`
- NFS e.g. `/mnt/nfs`
- HDFS e.g. `hdfs://<host>:<port>/<path>`

For more information, please visit [MLflow Artifact Stores](https://mlflow.org/docs/latest/tracking/artifacts-stores.html).

#### Use AWS S3 for Artifact Store

##### Option #1: Use AWS IAM Role ARN

Associate an AWS IAM role to the service account by adding annotations as follows:

```yaml
serviceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: "arn:aws:iam::account-id:role/<YOUR_IAM_ROLE_ARN>"
```

For detailed information, please visit [Configuring a Kubernetes service account to assume an IAM role - Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/associate-service-account-role.html).

##### Option #2: Read AWS Access Key From an Existing Secret

The existing secret should contain key `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, if you do not already have one, you can create a secret to store S3 access credentials as follows:

```bash
kubectl create secret generic mlflow-s3-artifact-store-secret \
    --from-literal=AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID> \
    --from-literal=AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>
```

Then configure `artifactStore.s3.existingSecret` to use the secret just created:

```yaml
artifactStore:
  s3:
    enabled: true
    existingSecret: mlflow-s3-artifact-store-secret
```

##### Option #3: Configure AWS Access Key Directly

If no existing credentials secret is specified, `artifactStore.s3.createSecret` can be directly configured to specify AWS access key, and a new secret will be created to hold it:

```yaml
artifactStore:
  s3:
    enabled: true
    createSecret:
      accessKeyId: <YOUR_AWS_ACCESS_KEY_ID>
      secretAccessKey: <YOUR_AWS_SECRET_ACCESS_KEY>
```

##### Extra S3 Configurations

To add S3 file upload extra arguments, you need to set `MLFLOW_S3_UPLOAD_EXTRA_ARGS` to a JSON object of key/value pairs. For example, if you want to upload to a KMS Encrypted bucket using the KMS Key 1234:

```yaml
extraEnv:
- name: MLFLOW_S3_UPLOAD_EXTRA_ARGS
  value: |
    {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "1234"}
```

To store artifacts in a custom S3 endpoint, you need to set `MLFLOW_S3_ENDPOINT_URL` environment variable as follows:

```yaml
extraEnv:
- name: MLFLOW_S3_ENDPOINT
  value: <YOUR_S3_ENDPOINT_URL>
```

If you want to disable TLS authentication, you can set `MLFLOW_S3_IGNORE_TLS` variable to `true`:

```yaml
extraEnv:
- name: MLFLOW_S3_IGNORE_TLS
  value: true
```

Additionally, if MinIO server is configured with non-default region, you should set `AWS_DEFAULT_REGION` variable:

```yaml
extraEnv:
- name: AWS_DEFAULT_REGION
  value: <YOUR_REGION>
```

#### Use Google Cloud Storage for Artifact Store

For detailed information, please visit [Use Google Cloud Storage as Artifact Store](https://mlflow.org/docs/latest/tracking/artifacts-stores.html#google-cloud-storage).

##### Option #1: Read Google Cloud Storage Access Credentials From an Existing Secret

The existing secret should contain key `keyfile.json`, if you do not already have one, you can create a secret from your `keyfile.json` as follows:

```yaml
kubectl create secret generic mlflow-gcp-artifact-store-secret \
    --from-file=keyfile.json
```

Then configure `artifactStore.gcp.existingSecret` to use the secret just created:

```yaml
artifactStore:
  gcp:
    enabled: true
    existingSecret: mlflow-gcp-artifact-store-secret
```

##### Option #2: Configure Google Cloud Storage Access Credentials Directly

If no existing secret is specified, `artifactStore.gcp.createSecret` can be configured to specify GCS access credentials directly:

```yaml
artifactStore:
  gcp:
    enabled: true
    createSecret:
      keyFile: <YOUR_KEY_FILE_CONTENT>
```

##### Extra GCS Configurations

You may set some MLflow environment variables to troubleshoot GCS read-timeouts by setting the following variables:

```yaml
extraEnv:
# Sets the standard timeout for transfer operations in seconds (Default: 60 for GCS). Use -1 for indefinite timeout.
- name: MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT
  value: 60
# Sets the standard upload chunk size for bigger files in bytes (Default: 104857600 ≙ 100MiB), must be multiple of 256 KB.
- name: MLFLOW_GCS_UPLOAD_CHUNK_SIZE
  value: 104857600
# Sets the standard download chunk size for bigger files in bytes (Default: 104857600 ≙ 100MiB), must be multiple of 256 KB.
- name: MLFLOW_GCS_DOWNLOAD_CHUNK_SIZE
  value: 104857600
```

#### Use Azure Blob Storage for Artifact Store

For detailed information, please visit [Use Azure Blob Storage as Artifact Store](https://mlflow.org/docs/latest/tracking/artifacts-stores.html#azure-blob-storage).

##### Option #1: Read Azure Access Credentials From an Existing Secret

The existing secret should contain key `AZURE_STORAGE_CONNECTION_STRING` and `AZURE_STORAGE_ACCESS_KEY`, if you do not already have one, you can create a secret as follows:

```yaml
kubectl create secret generic mlflow-azure-artifact-store-secret \
    --from-literal=AZURE_STORAGE_CONNECTION_STRING=<AZURE_STORAGE_CONNECTION_STRING> \
    --from-literal=AZURE_STORAGE_ACCESS_KEY=<YOUR_AZURE_STORAGE_ACCESS_KEY>
```

Then configure `artifactStore.azure.existingSecret` to use the secret just created:

```yaml
artifactStore:
  azure:
    enabled: true
    existingSecret: mlflow-azure-artifact-store-secret
```

##### Option #2: Configure Azure Access Credentials Directly

If no existing secret is specified, `artifactStore.azure.createSecret` can be configured directly to specify Azure access credentials:

```yaml
artifactStore:
  azure:
    enabled: true
    createSecret:
      azureStorageConnectionString: <YOUR_AZURE_STORAGE_CONNECTION_STRING>
      azureStorageAccessKey: <YOUR_AZURE_STORAGE_ACCESS_KEY>
```

#### Use Alibaba Cloud OSS for Artifact Store

The [aliyunstoreplugin](https://pypi.org/project/aliyunstoreplugin/) allows MLflow to use Alibaba Cloud OSS storage as an artifact store. There are two different ways to access OSS:

1. Associates an Alibaba Cloud RAM role to the service account **(Recommended)**.
2. Use Alibaba Cloud RAM user's AccessKey ID and AccessKey Secret.

##### Option #1: Use Alibaba Cloud RAM Role and RRSA

Associate an Alibaba Cloud RAM role to the service account by adding annotations as follows:

```yaml
serviceAccount:
  create: true
  annotations:
    pod-identity.alibabacloud.com/role-name: <YOUR_ALIBABA_CLOUD_RAM_ROLE_NAME>
```

For more information, please visit [Use RRSA to authorize different pods to access different cloud services - Container Service for Kubernetes - Alibaba Cloud Documentation Center](https://www.alibabacloud.com/help/ack/ack-managed-and-ack-dedicated/user-guide/use-rrsa-to-authorize-pods-to-access-different-cloud-services).

##### Option #2: Read Alibaba Cloud AccessKey From an Existing Secret

The existing secret should contain key `MLFLOW_OSS_KEY_ID` and `MLFLOW_OSS_KEY_SECRET`, if you do not already have one, you can create a secret to store Alibaba Cloud AccessKey ID and AccessKey secret as follows:

```bash
kubectl create secret generic mlflow-oss-artifact-store-secret \
    --from-literal=MLFLOW_OSS_KEY_ID=<ALIBABA_CLOUD_ACCESS_KEY_ID> \
    --from-literal=MLFLOW_OSS_KEY_SECRET=<ALIBABA_CLOUD_ACCESS_KEY_SECRET>
```

Then configure `artifactStore.oss.existingSecret` to use the secret just created:

```yaml
artifactStore:
  oss:
    enabled: true
    existingSecret: mlflow-oss-artifact-store-secret
```

##### Option #3: Configure Alibaba Cloud AccessKey Directly

If no existing secret is specified, you can configure `artifactStore.oss.createSecret` directly to specify Alibaba Cloud access credentials as follows and a new secret will be created to store it:

```yaml
artifactStore:
  oss:
    enabled: true
    createSecret:
      accessKeyId: <YOUR_ALIBABA_CLOUD_ACCESS_KEY_ID>
      accessKeySecret: <YOUR_ALIBABA_CLOUD_ACCESS_KEY_SECRET>
```

### Authentication

#### Use BasicAuth

MLflow supports basic HTTP authentication to enable access control over experiments and registered models.

Suppose you have `basic_auth.ini` file as follows:

```ini
[mlflow]
default_permission = READ
database_uri = sqlite:///basic_auth.db
admin_username = admin
admin_password = password
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
```

Create a secret to store basic auth configurations from configuration file:

```bash
kubectl create secret generic mlflow-basic-auth-secret --from-file=basic_auth.ini
```

Then enable BasicAuth and use the secret just created:

```yaml
trackingServer:
  basicAuth:
    enabled: true
    existingSecret: mlflow-basic-auth-secret
```

Otherwise, you can directly configure basic authentication as follows and a new secret will be created to hold it:

```yaml
trackingServer:
  basicAuth:
    enabled: true
    createSecret:
      defaultPermission: READ
      databaseUri: sqlite:///basic_auth.db
      adminUsername: admin
      adminPassword: password
      authorizationFunction: mlflow.server.auth:authenticate_request_basic_auth
```

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| affinity | object | `{}` | Pod affinity |
| artifactStore.azure.createSecret | object | `{"azureStorageAccessKey":"","azureStorageConnectionString":""}` | If Azure is enabled as artifact store backend and no existing secret is specified, create the secret used to connect to Azure |
| artifactStore.azure.enabled | bool | `false` | Specifies whether to enable Azure Blob Storage as artifact store backend |
| artifactStore.azure.existingSecret | string | `""` | Name of an existing secret containing the key `AZURE_STORAGE_CONNECTION_STRING` or `AZURE_STORAGE_ACCESS_KEY` to store credentials to access artifact storage on AZURE |
| artifactStore.gcp.createSecret | object | `{"keyFile":""}` | If GCP is enabled as artifact storage and no existing secret is specified, create the secret used to connect to GCP |
| artifactStore.gcp.createSecret.keyFile | string | `""` | Content of key file |
| artifactStore.gcp.enabled | bool | `false` | Specifies whether to enable Google Cloud Storage as artifact store backend |
| artifactStore.gcp.existingSecret | string | `""` | Name of an existing secret containing the key `keyfile.json` used to store credentials to access GCP |
| artifactStore.oss.createSecret | object | `{"accessKeyId":"","accessKeySecret":""}` | If OSS is enabled as artifact store backend and no existing secret is specified, create the secret used to store OSS access credentials |
| artifactStore.oss.createSecret.accessKeyId | string | `""` | Alibaba Cloud access key ID |
| artifactStore.oss.createSecret.accessKeySecret | string | `""` | Alibaba Cloud access key secret |
| artifactStore.oss.enabled | bool | `false` | Specifies whether to enable Alibaba Cloud Object Store Service(OSS) as artifact store backend |
| artifactStore.oss.endpoint | string | `""` | Endpoint of OSS e.g. oss-cn-beijing-internal.aliyuncs.com |
| artifactStore.oss.existingSecret | string | `""` | Name of an existing secret containing the key `MLFLOW_OSS_KEY_ID` and `MLFLOW_OSS_KEY_SECRET` to store credentials to access OSS |
| artifactStore.s3.createCaSecret | object | `{"caBundle":""}` | If S3 is enabled as artifact store backend and no existing CA secret is specified, create the secret used to secure connection to S3 / Minio |
| artifactStore.s3.createCaSecret.caBundle | string | `""` | Content of CA bundle |
| artifactStore.s3.createSecret | object | `{"accessKeyId":"","secretAccessKey":""}` | If S3 is enabled as artifact storage backend and no existing secret is specified, create the secret used to connect to S3 / Minio |
| artifactStore.s3.createSecret.accessKeyId | string | `""` | AWS access key ID |
| artifactStore.s3.createSecret.secretAccessKey | string | `""` | AWS secret access key |
| artifactStore.s3.enabled | bool | `false` | Specifies whether to enable AWS S3 as artifact store backend |
| artifactStore.s3.existingCaSecret | string | `""` | Name of an existing secret containing the key `ca-bundle.crt` used to store the CA certificate for TLS connections |
| artifactStore.s3.existingSecret | string | `""` | Name of an existing secret containing the keys `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to access artifact storage on AWS S3 or MINIO |
| backendStore.createSecret | object | `{"backendStoreUri":""}` | If no existing secret is specified, creates a secret to store backend store uri |
| backendStore.createSecret.backendStoreUri | string | `""` | Backend store uri |
| backendStore.databaseUpgrade | bool | `false` | Specifies whether to run `mlflow db upgrade ${MLFLOW_BACKEND_STORE_URI}` to upgrade database schema when use a database as backend store |
| backendStore.existingSecret | string | `""` | Name of an existing secret which contains key `MLFLOW_BACKEND_STORE_URI` |
| extraContainers | list | `[]` | Extra containers belonging to the mlflow pod. |
| extraEnv | list | `[]` | Extra environment variables in mlflow container |
| extraEnvFrom | list | `[]` | Extra environment variable sources in mlflow container |
| extraInitContainers | list | `[]` | Extra initialization containers belonging to the mlflow pod. |
| extraVolumeMounts | list | `[]` | Extra volume mounts to mount into the mlflow container's file system |
| extraVolumes | list | `[]` | Extra volumes that can be mounted by containers belonging to the mlflow pod |
| fullnameOverride | string | `""` | String to override the default generated fullname |
| image.pullPolicy | string | `"IfNotPresent"` | Image pull policy |
| image.pullSecrets | list | `[]` | Image pull secrets for private docker registry |
| image.registry | string | `"ghcr.io"` | Docker image registry |
| image.repository | string | `"mlflow/mlflow"` | Docker image repository |
| image.tag | string | `""` | Docker image tag, default is `v${appVersion}` |
| ingress.annotations | object | `{}` | Annotations to add to the ingress |
| ingress.className | string | `""` | Ingress class name |
| ingress.enabled | bool | `false` | Specifies whether a ingress should be created |
| ingress.hosts | list | `[{"host":"chart-example.local","paths":[{"path":"/","pathType":"ImplementationSpecific"}]}]` | Host rules to configure the ingress |
| ingress.tls | list | `[]` | TLS configuration |
| nameOverride | string | `""` | String to override the default generated name |
| nodeSelector | object | `{}` | Pod node selector |
| podAnnotations | object | `{}` | Pod annotations |
| podSecurityContext | object | `{}` | Pod security context |
| replicaCount | int | `1` | Number of mlflow server replicas to deploy |
| resources | object | `{}` | Pod resources |
| securityContext | object | `{}` | Container security context |
| service.annotations | object | `{}` | Annotations to add to the service |
| service.name | string | `"http"` | Service port name |
| service.port | int | `5000` | Service port number |
| service.type | string | `"ClusterIP"` | Specifies which type of service should be created |
| serviceAccount.annotations | object | `{}` | Annotations to add to the service account |
| serviceAccount.create | bool | `true` | Specifies whether a service account should be created |
| serviceAccount.name | string | `""` | Name of the service account to use. If not set and create is true, a name is generated using the fullname template |
| tolerations | list | `[]` | Pod tolerations |
| trackingServer.artifactsDestination | string | `""` | Specifies the base artifact location from which to resolve artifact upload/download/list requests (e.g. `s3://my-bucket`) |
| trackingServer.basicAuth.createSecret.adminPassword | string | `"password"` | Default admin password if the admin is not already created |
| trackingServer.basicAuth.createSecret.adminUsername | string | `"admin"` | Default admin username if the admin is not already created |
| trackingServer.basicAuth.createSecret.authorizationFunction | string | `"mlflow.server.auth:authenticate_request_basic_auth"` | Function to authenticate requests |
| trackingServer.basicAuth.createSecret.databaseUri | string | `"sqlite:///basic_auth.db"` | Database location to store permissions and user data |
| trackingServer.basicAuth.createSecret.defaultPermission | string | `"READ"` | Default permission on all resources |
| trackingServer.basicAuth.enabled | bool | `false` | Specifies whether to enable basic authentication |
| trackingServer.basicAuth.existingSecret | string | `""` | Name of an existing secret which contains key `basic_auth.ini` |
| trackingServer.defaultArtifactRoot | string | `""` | Specifies a default artifact location for logging, data will be logged to `mlflow-artifacts/:` if artifact serving is enabled, otherwise `./mlruns` |
| trackingServer.extraArgs | list | `[]` | Extra arguments passed to the `mlflow server` command |
| trackingServer.host | string | `"0.0.0.0"` | Network address to listen on |
| trackingServer.mode | string | `"no-serve-artifacts"` | Specifies which mode mlflow tracking server run with, available options are `serve-artifacts`, `no-serve-artifacts` and `artifacts-only` |
| trackingServer.port | int | `5000` | Port to expose the tracking server |
| trackingServer.workers | int | `1` | Number of gunicorn worker processes to handle requests |

## Source Code

* <https://github.com/mlflow/mlflow/tree/master/charts/mlflow>

## Maintainers

| Name | Email | Url |
| ---- | ------ | --- |
| Yi Chen | <github@chenyicn.net> |  |
