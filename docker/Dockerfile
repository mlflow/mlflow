FROM python:3.10-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV MLFLOW_HOME=/mlflow
ENV MLFLOW_DATA_DIR=/mlflow/mlruns
WORKDIR $MLFLOW_HOME

# Install MLflow
ARG MLFLOW_VERSION=1.30.0
RUN pip install --no-cache-dir mlflow==$MLFLOW_VERSION

# Copy entry script
COPY start-mlflow.sh /start-mlflow.sh
RUN chmod +x /start-mlflow.sh

# Expose MLflow UI port
EXPOSE 5000

CMD ["/start-mlflow.sh"]
