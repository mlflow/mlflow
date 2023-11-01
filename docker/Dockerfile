FROM python:3.10-slim-bullseye
ARG VERSION
RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache mlflow==$VERSION
