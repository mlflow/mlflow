FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV MLFLOW_HOME=/mlflow
ENV MLFLOW_DATA_DIR=/mlflow/mlruns
WORKDIR $MLFLOW_HOME

ARG MLFLOW_VERSION=1.30.0
RUN pip install --no-cache-dir mlflow==$MLFLOW_VERSION

# Copy Python startup script
COPY start_mlflow.py /start_mlflow.py

# Expose UI port
EXPOSE 5000

CMD ["python", "/start_mlflow.py"]
