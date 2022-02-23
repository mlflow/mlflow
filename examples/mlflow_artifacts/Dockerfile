FROM python:3.7

WORKDIR /app

# Install mlflow and packages requied to interact with PostgreSQL and MinIO
RUN pip install psycopg2 boto3 git+https://www.github.com/mlflow/mlflow@proxy-artifact-access-fix
