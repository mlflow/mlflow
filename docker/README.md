# MLflow Docker Images

MLflow provides Docker images to help you quickly deploy and run MLflow in containerized environments.

## Image Variants

### mlflow:VERSION (default)

This image contains only the core MLflow package without extra dependencies. Most integrations (backend store databases, artifact stores, etc.) will not work without additional packages.

Use this image as a lightweight base when you want full control over which dependencies to install, or when you only need basic MLflow functionality.

### mlflow:VERSION-full

This image contains MLflow with all extra dependencies, including:

- Database drivers (MySQL, PostgreSQL, SQL Server)
- Cloud storage integrations (AWS S3, Azure Blob, GCS)
- AI Gateway and GenAI capabilities
- Databricks integrations

Use this image when you need comprehensive MLflow functionality with multiple integrations.

**Note:** Replace `VERSION` with the actual MLflow version (e.g., `3.9.0`) or use `latest-full` for the most recent release.

## Quick Start

### Basic Usage

Run MLflow server with default settings (SQLite backend, local file storage):

```bash
docker run -p 5000:5000 mlflow:latest-full mlflow server --host 0.0.0.0
```

Access the MLflow UI at http://localhost:5000

### With MySQL Backend

```bash
docker run -p 5000:5000 \
  -e MLFLOW_BACKEND_STORE_URI=mysql+pymysql://user:password@mysql-host:3306/mlflow \
  mlflow:latest-full \
  mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_URI --host 0.0.0.0
```

### With PostgreSQL Backend

```bash
docker run -p 5000:5000 \
  -e MLFLOW_BACKEND_STORE_URI=postgresql://user:password@postgres-host:5432/mlflow \
  mlflow:latest-full \
  mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_URI --host 0.0.0.0
```

### With S3 Artifact Storage

```bash
docker run -p 5000:5000 \
  -e AWS_ACCESS_KEY_ID=your-access-key \
  -e AWS_SECRET_ACCESS_KEY=your-secret-key \
  mlflow:latest-full \
  mlflow server --artifacts-destination s3://your-bucket/path --host 0.0.0.0
```

### With Azure Blob Storage

```bash
docker run -p 5000:5000 \
  -e AZURE_STORAGE_CONNECTION_STRING="your-connection-string" \
  mlflow:latest-full \
  mlflow server --artifacts-destination wasbs://container@account.blob.core.windows.net/path --host 0.0.0.0
```

## Docker Compose Example

Here's an example `docker-compose.yml` for running MLflow with MySQL:

```yaml
version: "3.8"

services:
  mysql:
    image: mysql:8
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: mlflow
      MYSQL_USER: mlflow
      MYSQL_PASSWORD: mlflow
    volumes:
      - mysql-data:/var/lib/mysql
    ports:
      - "3306:3306"

  mlflow:
    image: mlflow:latest-full
    depends_on:
      - mysql
    ports:
      - "5000:5000"
    environment:
      MLFLOW_BACKEND_STORE_URI: mysql+pymysql://mlflow:mlflow@mysql:3306/mlflow
    command: mlflow server --backend-store-uri $MLFLOW_BACKEND_STORE_URI --host 0.0.0.0

volumes:
  mysql-data:
```

## Environment Variables

Common environment variables for configuring MLflow:

- `MLFLOW_BACKEND_STORE_URI` - Backend store URI (database connection string)
- `MLFLOW_DEFAULT_ARTIFACT_ROOT` - Default location for storing artifacts
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - AWS credentials for S3
- `AZURE_STORAGE_CONNECTION_STRING` - Azure storage connection string
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP service account key file

## Running the Development Version

### Build the dev image

From the repository root:

```bash
docker build -f docker/Dockerfile.full.dev -t mlflow-dev .
```

This installs MLflow in editable mode with all extras: `[extras,db,databricks,gateway,genai,sqlserver]`

### Run the dev image

```bash
docker run -p 5000:5000 mlflow-dev mlflow server --host 0.0.0.0
```

**Note:** The dev Docker image is intended for testing backend changes only, not for production use.

## Building Custom Images

If you need to customize the image, you can use the base image and add your own dependencies:

```dockerfile
FROM mlflow:latest

# Install additional dependencies
RUN pip install mlflow[extras,db] your-custom-package

# Add custom configurations
COPY your-config.yaml /opt/mlflow/
```

Or start from the full image and add more:

```dockerfile
FROM mlflow:latest-full

# Install additional custom packages
RUN pip install your-custom-package
```
