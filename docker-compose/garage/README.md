
# MLflow + GarageHQ (PostgreSQL + Garage S3)

[GarageHQ](https://garagehq.deuxfleurs.fr/) is a distributed, S3-compatible object store. This setup runs:

- `postgres`: MLflow backend store
- `garage`  : S3-compatible object storage
- `create-bucket`: idempotent bucket bootstrap
- `mlflow`  : MLflow Tracking Server

> **Why Garage?**  
> Lightweight, distributed, ideal to test S3 behaviours in a peer-to-peer style deployment.

---

## Quick Start

1. Copy env and adjust values:
   ```bash
   cp .env.example .env
   ```
2. Change in the `garage.toml`:
   1. `rpc_secret`
   2. `admin_token`
   3. `metrics_token`
3. Start services:
   ```bash
   docker compose up -d --build
   ```
4. Open MLflow: `http://localhost:${MLFLOW_PORT}`

---

## Environment Variables (example)

**Common variables** :

- **PostgreSQL**
  - `POSTGRES_USER=mlflow`
  - `POSTGRES_PASSWORD=mlflow`
  - `POSTGRES_DB=mlflow`

- **S3**
  - `ACCESS_KEY_ID=admin`
  - `DEFAULT_REGION=us-east-1`
  - `S3_BUCKET=mlflow`

- **MLFlow**
  - `MLFLOW_VERSION=latest`
  - `MLFLOW_HOST=0.0.0.0`
  - `MLFLOW_PORT=5000`
  - `MLFLOW_BACKEND_STORE_URI=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}`
  - `MLFLOW_ARTIFACTS_DESTINATION=s3://${S3_BUCKET}`
  - `MLFLOW_S3_ENDPOINT_URL=http://storage:9000`

> Use the exact endpoint and port defined in this folder's `docker-compose.yml`.

---

## Bucket Bootstrap (idempotent)

GarageHQ uses a different method to create the keys and then create a bucket because the mechanism is different
from the other S2 compatible stores.

The script `garage-init.sh` creates the keys and then stores it in the common volume only if the keys are not already there.

If the keys are already in the volume skips the creation of the keys and the creation of the bucket.

---

## Verify

```bash
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3api list-buckets

echo hi > /tmp/t.txt
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3 cp /tmp/t.txt s3://${S3_BUCKET}/t.txt
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3 cp s3://${S3_BUCKET}/t.txt -
```

---

## Troubleshooting

- Ensure `MLFLOW_S3_ENDPOINT_URL` uses the **compose service name** (not localhost) from MLflow container perspective.
- If AWS CLI errors on create-bucket due to permissions/policy, pre-create the bucket via Garage admin/CLI and keep the `head-bucket` gate to remain idempotent.
