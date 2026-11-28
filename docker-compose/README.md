# MLflow with Docker Compose (PostgreSQL + S3-Compatible Storage)

This directory provides a **Docker Compose** setup for running **MLflow** locally with a **PostgreSQL** backend store and **RustFS** for S3-compatible artifact storage.

---

## Overview

- **MLflow Tracking Server**  
  Serves the REST API and UI (default: `http://localhost:5000`).

- **PostgreSQL**  
  Stores MLflow's metadata (experiments, runs, params, metrics).

- **RustFS Artifact Storage**  
  Stores model files and run artifacts.

---

## Prerequisites

- **Git**
- **Docker** and **Docker Compose**
- On macOS/Windows: Docker Desktop
- On Linux: Docker Engine + compose plugin

Verify installation:

```bash
docker --version
docker compose version
```

---

## 1. Clone the Repository

```bash
git clone https://github.com/mlflow/mlflow.git
cd mlflow/docker-compose/
```

---

## 2. Configure Environment

Copy and customize the environment file:

```bash
cp .env.dev.example .env
```

The `.env` file defines:

- MLflow server port
- PostgreSQL credentials
- S3 bucket name
- S3-compatible endpoint URL
- Backend-specific configuration for RustFS

**Common variables**:

- **PostgreSQL**

  - `POSTGRES_USER=mlflow`
  - `POSTGRES_PASSWORD=mlflow`
  - `POSTGRES_DB=mlflow`

- **S3**

  - `AWS_ACCESS_KEY_ID=s3admin`
  - `AWS_SECRET_ACCESS_KEY=s3admin`
  - `AWS_DEFAULT_REGION=us-east-1`
  - `S3_BUCKET=mlflow`

- **RustFS**

  - `RUSTFS_CONSOLE_ENABLE=true`

- **MLflow**
  - `MLFLOW_VERSION=latest`
  - `MLFLOW_HOST=0.0.0.0`
  - `MLFLOW_PORT=5000`
  - `MLFLOW_BACKEND_STORE_URI=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}`
  - `MLFLOW_ARTIFACTS_DESTINATION=s3://${S3_BUCKET}`
  - `MLFLOW_S3_ENDPOINT_URL=http://storage:9000`

---

## 3. Launch the Stack

Inside directory **mlflow/docker-compose**:

```bash
docker compose up -d
```

This will:

- Start PostgreSQL
- Start RustFS
- Start MLflow
- Create the S3 bucket if it doesn't exist

Check status:

```bash
docker compose ps
```

Tail logs:

```bash
docker compose logs -f
```

---

## 4. Access MLflow

Once running:

- Open `http://localhost:5000` (or the port defined in `.env`)

You can now log runs, metrics, artifacts, and models to your local MLflow instance.

---

## 5. Shutdown

To stop and remove containers:

```bash
docker compose down
```

To reset everything, including volumes:

```bash
docker compose down -v
```

---

## Tips & Troubleshooting

### RustFS Notes (important)

- Set **server domains/host** so virtual-hosted requests can be resolved by RustFS:

  ```env
  RUSTFS_SERVER_DOMAINS=storage:9000
  ```

  (match the compose service DNS name)

- Prefer AWS CLI **`s3api`** for bucket creation. Some S3 clients default to **path-style** on custom endpoints; if bucket creation fails with `InvalidBucketName`, switch to `s3api` or a client like MinIO `mc`.

- Inside MLflow, use the internal endpoint:
  ```env
  MLFLOW_S3_ENDPOINT_URL=http://storage:9000
  MLFLOW_ARTIFACTS_DESTINATION=s3://mlflow/
  ```

### Healthcheck Example

RustFS usually responds on `/health` with a json that contains the status of the server:

```sh
curl -s http://127.0.0.1:9000/health | grep -q '\"status\"\\s*:\\s*\"ok\"'
```

Use that in a container healthcheck (no `-f`, 4xx may appear during bootstrap).

---

### Artifact Upload Issues

Verify:

- `MLFLOW_ARTIFACTS_DESTINATION=s3://<bucket>/`
- `MLFLOW_S3_ENDPOINT_URL=http://<service>:<port>`
- AWS credentials match the backend configuration

To verify S3 storage is working:

```bash
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3api list-buckets

echo hi > /tmp/t.txt
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3 cp /tmp/t.txt s3://${S3_BUCKET}/t.txt
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3 cp s3://${S3_BUCKET}/t.txt -
```

If this passes, MLflow can read and write artifacts to RustFS.

### Troubleshooting

- `InvalidBucketName` on create-bucket → use `s3api` (virtual-host friendly) or MinIO `mc`; ensure `RUSTFS_SERVER_DOMAINS` matches the S3 hostname.
- Endpoint issues from MLflow → make sure `MLFLOW_S3_ENDPOINT_URL` uses the **service name** visible from MLflow (e.g., `http://storage:9000`).

### Resetting the Environment

```bash
docker compose down -v
docker compose up -d
```

### Logs

```bash
docker compose logs -f mlflow
docker compose logs -f postgres
docker compose logs -f storage
```

### Port Conflicts

Edit `.env` and restart containers:

```bash
docker compose down
docker compose up -d
```

---

## Next Steps

- Point your training scripts to this server:
  ```bash
  export MLFLOW_TRACKING_URI=http://localhost:5000
  ```
- Start logging runs with `mlflow.start_run()` (Python) or the MLflow CLI.
- Customize the `.env` and `docker-compose.yml` to fit your local workflow (e.g., change image tags, add volumes, etc.).

---

**You now have a fully local MLflow stack with persistent metadata and artifact storage—ideal for development and experimentation.**

## OIDC Authentication

[`docker-compose.oidc.yml`](docker-compose.oidc.yml) runs the official MLflow full
image with the community-maintained
[`mlflow-oidc-auth`](https://github.com/mlflow-oidc/mlflow-oidc-auth) plugin. This
standalone example uses separate SQLite files for tracking and authentication,
and local artifact storage in a named volume. It does not require the PostgreSQL
or storage services in the default Compose file.

### Configure an identity provider

Register a confidential OIDC client with your identity provider. Set its redirect
URI to `http://localhost:5000/callback`, allow the `openid`, `email`, and `profile`
scopes, and include a `groups` claim in the ID token. Create an ordinary user in
`example-users` and an administrator in `example-admins`, or change the group
names in the environment file to match your provider.

Copy the example configuration:

```bash
cp .env.oidc.example .env.oidc
chmod 600 .env.oidc
python -c 'import secrets; print(secrets.token_hex(32))'
```

Edit `.env.oidc`: set `OIDC_DISCOVERY_URL`, `OIDC_CLIENT_ID`, and
`OIDC_CLIENT_SECRET` from the provider, and set `SECRET_KEY` to the generated
value. Keep this key across restarts and keep the environment file out of version
control. The discovery, token, and user information endpoints must be reachable
from the container; `localhost` inside the container refers to the container
itself. See the [SSO documentation](https://mlflow.org/docs/latest/self-hosting/security/sso/)
for provider-specific setup.

### Start and verify

Run these commands from this directory:

```bash
docker compose --env-file .env.oidc -f docker-compose.oidc.yml config --quiet
docker compose --env-file .env.oidc -f docker-compose.oidc.yml up -d
docker compose --env-file .env.oidc -f docker-compose.oidc.yml logs -f mlflow
```

Open `http://localhost:5000` and sign in with the administrator account. Use the
plugin's permission UI to grant an ordinary user access to the required
experiments. Group membership permits sign-in; it does not by itself grant
access to every experiment. Follow the [plugin's client authentication instructions](https://github.com/mlflow-oidc/mlflow-oidc-auth/blob/main/docs/programmatic-access.md)
to configure SDK credentials. An HTTP health check only verifies server
availability; also verify sign-in and access with both accounts.

Permission changes can take up to the plugin's permission-cache TTL (30 seconds
by default) to become visible across server workers.

The plugin is installed at startup, so the container needs package-index access.
The install pins MLflow to the image's version and fails if the selected plugin
is incompatible, rather than silently upgrading MLflow. Review both version
pins together before changing them.

### Stop, restart, and upgrade

```bash
docker compose --env-file .env.oidc -f docker-compose.oidc.yml down
docker compose --env-file .env.oidc -f docker-compose.oidc.yml up -d
```

The named volume preserves tracking data, authentication data, and artifacts.
Do not add `--volumes` to `down` unless you intend to delete them. Back up the
volume and stop all writers before upgrading. Follow the
[MLflow database upgrade instructions](https://mlflow.org/docs/latest/self-hosting/migration/)
and the plugin's upgrade instructions for their respective databases.

This example binds only to the host's loopback interface and is intended for
local evaluation. Before exposing it remotely, configure HTTPS, the public
callback URL, allowed hosts, and trusted proxies using the
[plugin configuration reference](https://github.com/mlflow-oidc/mlflow-oidc-auth/blob/main/docs/configuration.md).
For external artifact storage, see the
[artifact store documentation](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/).
