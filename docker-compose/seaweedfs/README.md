
# MLflow + SeaweedFS (PostgreSQL + SeaweedFS S3 Gateway)

SeaweedFS provides a distributed filesystem with an **S3 gateway**. We have two setups:

- `docker-compose.simple.yml`: Simple setup with all-in-one service of Seaweedfs
- `docker-compose.multi.yml`: Setup with an example of a more complex example of Seaweedfs setup

## Simple setup configuration

- `postgres`: MLflow backend store
- `storage`: all-in-one Seaweedfs cluster
- `create-bucket`: idempotent bucket bootstrap
- `mlflow`: MLflow Tracking Server

## Multi setup configuration

- `postgres`: MLflow backend store
- `seaweedfs-master`: cluster coordinator
- `seaweedfs-volume-1` / `seaweedfs-volume-2`: data nodes
- `seaweedfs-filer`: metadata and file view (HTTP)
- `seaweedfs-s3`: S3-compatible gateway
- `create-bucket`: idempotent bucket bootstrap
- `mlflow`: MLflow Tracking Server

---

## Quick Start

1. Copy env and adjust values:
   ```bash
   cp .env.dev.example .env
   ```
2. Start services:
   1. Simple setup
      ```bash
      docker compose -f docker-compose.simple.yml up -d
      ```
   2. Multi setup
      ```bash
      docker compose -f docker-compose.multi.yml up -d
      ```


3. Open:
   - MLflow: `http://localhost:${MLFLOW_PORT}`
   - SeaweedFS Master UI: `http://localhost:9333/`
   - Filer UI: `http://localhost:8888/`

---

## Important Settings

### Replication
SeaweedFS replication is set via **replica placement**:

- `000` – no replication (dev/single node per rack)
- `001` – 2 copies in the **same rack** (requires ≥2 volume servers in the same rack)
- `010` – 2 copies across different racks (requires ≥2 racks)
- `100` – 3 copies across different datacenters (requires ≥3 DCs)

For this compose, **two volume servers in the same rack** are used so you can set:

- Master: `-defaultReplication=001`
- Filer:  `-defaultReplicaPlacement=001`

> If you run one server per rack/DC, use `000` to avoid `Not enough data nodes` errors.

### Healthchecks (reliable)

- **Master**: `GET /cluster/status` → 200 OK
- **Filer**:  `GET /status` → 200 OK
- **S3**:     `HEAD /` (no `-f`, some responses are 403/405 which are fine during bootstrap)
- **Volumes**: use **container name** in URL (not `localhost`) for UI GET checks, or bind to `0.0.0.0`:
  - Volume-1: `GET http://seaweedfs-volume-1:8080/ui/index.html`
  - Volume-2: `GET http://seaweedfs-volume-2:8081/ui/index.html`

### Bucket Bootstrap (idempotent)

The `create-bucket` job should:

1) `head-bucket`  
2) If missing, ensure the Filer path `/buckets/<bucket>` exists  
3) `create-bucket`  

Example entrypoint:

```sh
set -e
if aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3api head-bucket --bucket ${S3_BUCKET} 2>/dev/null; then
  echo "Bucket exists"
else
  echo "Ensuring /buckets/${S3_BUCKET} on filer"
  echo "bootstrap" | curl -s -X POST --data-binary @- "http://seaweedfs-filer:8888/buckets/${S3_BUCKET}/.init" >/dev/null || true
  curl -s -X DELETE "http://seaweedfs-filer:8888/buckets/${S3_BUCKET}/.init" >/dev/null || true
  echo "Creating bucket ${S3_BUCKET}"
  aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3api create-bucket --bucket ${S3_BUCKET} --region ${AWS_DEFAULT_REGION}
fi
```

> This avoids the error: `collection exists but bucket directory missing`.

---

## Environment Variables

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

- **MLFlow**
  -  `MLFLOW_VERSION=latest`
  - `MLFLOW_HOST=0.0.0.0`
  - `MLFLOW_PORT=5000`
  - `MLFLOW_BACKEND_STORE_URI=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}`
  - `MLFLOW_ARTIFACTS_DESTINATION=s3://${S3_BUCKET}`
  - `MLFLOW_S3_ENDPOINT_URL=http://seaweedfs-s3:9000`

> Match `MLFLOW_S3_ENDPOINT_URL` to the **S3 gateway service name** inside the compose network.

---

## Verify

```bash
# list buckets
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3api list-buckets

# put/get object
echo hi > /tmp/t.txt
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3 cp /tmp/t.txt s3://${S3_BUCKET}/t.txt
aws --endpoint-url=${MLFLOW_S3_ENDPOINT_URL} s3 cp s3://${S3_BUCKET}/t.txt -
```

Then log an artifact via MLflow and check it appears in the Filer UI.

---

## Troubleshooting

- `Not enough data nodes` → use replica `000` **or** run multiple volumes in the same rack for `001`.
- Volume healthcheck never healthy on `localhost` → use service DNS name or start volumes with `-ip.bind=0.0.0.0`.
- `BucketAlreadyExists / bucket directory missing` → create `/buckets/<bucket>` on the Filer before `create-bucket` (see snippet above).
