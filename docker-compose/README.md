# MLflow with Docker Compose (PostgreSQL + Local Artifact Storage)

This directory provides a **Docker Compose** setup for running **MLflow** locally with a **PostgreSQL** backend store and **local file system** artifact storage. It's intended for quick evaluation and local development.

---

## Overview

- **MLflow Tracking Server** — exposed on your host (default `http://localhost:5000`).
- **PostgreSQL** — persists MLflow's metadata (experiments, runs, params, metrics).
- **Local Artifact Storage** — artifacts are stored in `./mlartifacts` on your host machine (mounted into the container).

Compose automatically reads configuration from a local `.env` file in this directory.

---

## Prerequisites

- **Git**
- **Docker** and **Docker Compose**
  - Windows/macOS: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Linux: Docker Engine + the `docker compose` plugin

Verify your setup:

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

Copy the example environment file and modify as needed:

```bash
cp .env.dev.example .env
```

The `.env` file defines container image tags, ports, and storage configuration. Open it and review values before starting the stack.

**Common variables**:

- **MLflow**
  - `MLFLOW_PORT=5000` — host port for the MLflow UI/API
  - `MLFLOW_ARTIFACTS_DESTINATION=file:///mlflow/mlartifacts` — artifacts stored locally
- **PostgreSQL**
  - `POSTGRES_USER=mlflow`
  - `POSTGRES_PASSWORD=mlflow`
  - `POSTGRES_DB=mlflow`

---

## 3. Launch the Stack

```bash
docker compose up -d
```

This:

- Builds/pulls images as needed
- Creates a user-defined network
- Starts **postgres** and **mlflow** containers

Check status:

```bash
docker compose ps
```

View logs (useful on first run):

```bash
docker compose logs -f
```

---

## 4. Access MLflow

Open the MLflow UI:

- **URL**: `http://localhost:5000` (or the port set in `.env`)

You can now create experiments, run training scripts, and log metrics, parameters, and artifacts to this local MLflow instance.

Artifacts will be stored in the `./mlartifacts` directory on your host machine.

---

## 5. Shutdown

To stop and remove the containers and network:

```bash
docker compose down
```

> Data is preserved in Docker **volumes** and the `./mlartifacts` directory. To remove volumes as well (irreversible), run:
>
> ```bash
> docker compose down -v
> ```

---

## Tips & Troubleshooting

- **Artifact storage**
  Artifacts are stored in `./mlartifacts` on your host. This directory is created automatically on first run.

- **Resetting the environment**
  If you want a clean slate, stop the stack and remove volumes and artifacts:

  ```bash
  docker compose down -v
  rm -rf ./mlartifacts
  docker compose up -d
  ```

- **Logs**

  - MLflow server: `docker compose logs -f mlflow`
  - PostgreSQL: `docker compose logs -f postgres`

- **Port conflicts**
  If `5000` is in use, change it in `.env` and restart:
  ```bash
  docker compose down
  docker compose up -d
  ```

---

## How It Works (at a Glance)

- MLflow uses **PostgreSQL** as the _backend store_ for experiment/run metadata.
- MLflow uses the **local filesystem** (`./mlartifacts`) as the _artifact store_.
- Docker Compose wires services on a shared network; MLflow talks to PostgreSQL by container name (e.g., `postgres`).

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
