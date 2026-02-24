
# MLflow + RustFS (PostgreSQL + RustFS S3-Compatible Storage)

[RustFS](https://docs.rustfs.com/) is a lightweight, S3-compatible storage server written in Rust. This setup runs:

- `postgres`: MLflow backend store
- `storage` : RustFS server (S3-compatible)
- `create-bucket`: idempotent bucket bootstrap
- `mlflow`  : MLflow Tracking Server

---

## Quick Start

1. Copy env and adjust values:
   ```bash
   cp .env.example .env
   ```
2. Start services:
   ```bash
   docker compose up -d
   ```
3. Open MLflow: `http://localhost:${MLFLOW_PORT}`

---

## RustFS Notes (important)

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

## Bucket Bootstrap (idempotent)

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

---

## Environment Variables (example)

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
- **MLFlow**
  - `MLFLOW_VERSION=latest`
  - `MLFLOW_HOST=0.0.0.0`
  - `MLFLOW_PORT=5000`
  - `MLFLOW_BACKEND_STORE_URI=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}`
  - `MLFLOW_ARTIFACTS_DESTINATION=s3://${S3_BUCKET}`
  - `MLFLOW_S3_ENDPOINT_URL=http://storage:9000`

---

## Verify

```bash
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3api list-buckets

echo hi > /tmp/t.txt
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3 cp /tmp/t.txt s3://${S3_BUCKET}/t.txt
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3 cp s3://${S3_BUCKET}/t.txt -
```

If this passes, MLflow can read and write artifacts to RustFS.

---

## Troubleshooting

- `InvalidBucketName` on create-bucket → use `s3api` (virtual-host friendly) or MinIO `mc`; ensure `RUSTFS_SERVER_DOMAINS` matches the S3 hostname.
- Endpoint issues from MLflow → make sure `MLFLOW_S3_ENDPOINT_URL` uses the **service name** visible from MLflow (e.g., `http://storage:9000`).
