FROM python:3.6

WORKDIR /app

# Install packages requied to interact with PostgreSQL and MinIO
RUN pip install psycopg2 boto3
# Cache mlflow's dependencies to speed up image rebuild
RUN pip install mlflow
# Reinstall mlflow via wheel
COPY dist ./dist
RUN pip install dist/mlflow-*.whl
