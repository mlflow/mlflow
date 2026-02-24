
# MLflow with Docker Compose (PostgreSQL + S3-Compatible Storage)

This directory provides a **Docker Compose** setup for running **MLflow** locally with a **PostgreSQL** backend store and an S3-compatible artifact stores like **RustFS**.

Originally this stack included **MinIO**, but the repository now uses **RustFS**.

---

## Overview

All variants share the same architecture:

- **MLflow Tracking Server**  
  Serves the REST API and UI (default: `http://localhost:5000`).

- **PostgreSQL**  
  Stores MLflow’s metadata (experiments, runs, params, metrics).

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
cd docker-compose
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

Example variables:

**Common variables** :

- **PostgreSQL**
  - `POSTGRES_USER=mlflow`
  - `POSTGRES_PASSWORD=mlflow`
  - `POSTGRES_DB=mlflow`

- **S3**
  - `AWS_ACCESS_KEY_ID=admin`
  - `AWS_SECRET_ACCESS_KEY=admin`
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

From **this directory**:

```bash
docker compose up -d
```

This will:

- Start PostgreSQL  
- Start the selected S3-compatible backend  
- Start MLflow  
- Optionally create the S3 bucket (backend-specific init job)

Check status:

```bash
docker compose ps
```

Tail logs:

```bash
docker compose logs -f
```

---

## 5. Access MLflow

Once running:

- Open `http://localhost:5000` (or the port defined in `.env`)

You can now log runs, metrics, artifacts, and models to your local MLflow instance.

---

## 6. Shutdown

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

RustFS usually responds on `/`:
```sh
curl -I http://storage:9000/
```
Use that in a container healthcheck (no `-f`, 4xx may appear during bootstrap).

---

### Bucket Bootstrap (idempotent)

```sh
set -e
if aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3api head-bucket --bucket ${S3_BUCKET} 2>/dev/null; then
  echo "Bucket exists"
else
  aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3api create-bucket --bucket ${S3_BUCKET} --region ${AWS_DEFAULT_REGION}
fi
```

> If `s3api create-bucket` still fails due to addressing quirks, use MinIO `mc`:
> ```sh
> mc alias set local http://storage:9000 ${AWS_ACCESS_KEY_ID} ${AWS_SECRET_ACCESS_KEY}
> mc mb --ignore-existing local/${S3_BUCKET}
> ```

### Artifact Upload Issues
Verify:

- `MLFLOW_ARTIFACTS_DESTINATION=s3://<bucket>/`
- `MLFLOW_S3_ENDPOINT_URL=http://<service>:<port>`
- AWS credentials match the backend configuration

Bash command useful to check if the S3 is working correctly

```bash
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3api list-buckets

echo hi > /tmp/t.txt
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3 cp /tmp/t.txt s3://${S3_BUCKET}/t.txt
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3 cp s3://${S3_BUCKET}/t.txt -
```

If this passes, MLflow can read and write artifacts to RustFS.
---

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
docker compose logs -f <s3-service>
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