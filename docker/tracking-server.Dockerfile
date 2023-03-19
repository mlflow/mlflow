FROM python:3.10-slim-bullseye
ARG VERSION
RUN pip install --no-cache \
        mlflow==$VERSION \
        boto3==1.26.60 \
        azure-storage-blob>=12.0.0 \
        azure-identity>=1.6.1 \
        google-cloud-storage==2.7.0 \
        prometheus-flask-exporter \
        psycopg2-binary==2.9.5
