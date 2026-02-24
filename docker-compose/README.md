
# MLflow with Docker Compose (PostgreSQL + S3-Compatible Storage)

This directory provides multiple **Docker Compose** setups for running **MLflow** locally with a **PostgreSQL** backend store and one of several **S3-compatible artifact stores**.

Originally this stack included **MinIO**, but the repository now supports **three additional S3-compatible backends**:

- **MinIO** (default)
- **GarageHQ** (distributed S3)
- **SeaweedFS** (distributed filesystem + S3 Gateway)
- **RustFS** (lightweight S3-compatible storage in Rust)

Each backend has its own subdirectory and README.

---

## Overview

All variants share the same architecture:

- **MLflow Tracking Server**  
  Serves the REST API and UI (default: `http://localhost:5000`).

- **PostgreSQL**  
  Stores MLflow’s metadata (experiments, runs, params, metrics).

- **S3-Compatible Artifact Storage**  
  Stores model files and run artifacts. Four implementations are available:

  | Backend      | Directory        | Notes                                      |
  |--------------|------------------|--------------------------------------------|
  | **MinIO**    | `./minio/`       | Default, simple, single-node.              |
  | **GarageHQ** | `./garage/`      | Distributed, persistent, peer-to-peer.     |
  | **SeaweedFS**| `./seaweedfs/`   | Distributed FS with S3 gateway.            |
  | **RustFS**   | `./rustfs/`      | Simple, fast, ideal for local development. |

All setups load configuration from an `.env` file inside the selected directory (see instructions below).

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

## 2. Choose an S3 Backend

Each backend has its own folder:

```text
docker-compose/
 ├── minio/
 ├── garage/
 ├── seaweedfs/
 └── rustfs/
```

Enter the directory of the backend you want to use.

### Example (MinIO):

```bash
cd minio
```

### Example (SeaweedFS):

```bash
cd seaweedfs
```

Each directory contains:

- its own `docker-compose.yml`
- its own `.env.example`
- a dedicated `README.md` explaining backend-specific configuration

---

## 3. Configure Environment

Copy and customize the environment file inside the chosen backend directory:

```bash
cp .env.example .env
```

The `.env` file defines:

- MLflow server port  
- PostgreSQL credentials  
- S3 bucket name  
- S3-compatible endpoint URL  
- Backend-specific configuration (MinIO, GarageHQ, SeaweedFS, RustFS)

Example variables:

```env
MLFLOW_PORT=5000
MLFLOW_ARTIFACTS_DESTINATION=s3://mlflow/
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
POSTGRES_DB=mlflow
```

Each backend README documents the correct endpoint URL and required variables.

---

## 4. Launch the Stack

From **inside the backend directory**:

```bash
docker compose up -d
```

or for GarageHQ

```bash
docker compose up -d --build
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

### Artifact Upload Issues
Verify:

- `MLFLOW_ARTIFACTS_DESTINATION=s3://<bucket>/`
- `MLFLOW_S3_ENDPOINT_URL=http://<service>:<port>`
- AWS credentials match the backend configuration

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

## Backend-Specific READMEs

- **minio/README.md** — simple S3 dev setup  
- **garage/README.md** — distributed S3-compatible object storage  
- **seaweedfs/README.md** — SeaweedFS master/volume/filer/S3 gateway architecture  
- **rustfs/README.md** — lightweight Rust-based S3-compatible local storage  

These explain:

- architecture  
- compose layout  
- healthchecks  
- bucket initialization  
- known limitations and troubleshooting  

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