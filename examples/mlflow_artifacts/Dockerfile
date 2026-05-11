FROM python:3.10-slim

WORKDIR /app

# Install mlflow and packages required to interact with PostgreSQL and MinIO
RUN pip install --no-cache-dir mlflow psycopg2-binary boto3
