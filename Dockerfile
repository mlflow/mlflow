FROM python:3.10-bullseye

# Set environment variables
ENV PATH="/home/mlflow/.local/protoc/bin:$PATH"
ENV MLFLOW_DATA_DIR=/home/mlflow/mlruns

WORKDIR /home/mlflow

# Install Node.js, Java, Yarn, and Protoc
RUN curl -sL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends \
        nodejs \
        openjdk-11-jre-headless \
    && npm install --global yarn \
    && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.19.4/protoc-3.19.4-linux-x86_64.zip -O /tmp/protoc.zip \
    && mkdir -p /home/mlflow/.local \
    && unzip /tmp/protoc.zip -d /home/mlflow/.local/protoc \
    && rm /tmp/protoc.zip \
    && chmod -R +x /home/mlflow/.local/protoc \
    # Create unprivileged user
    && groupadd --gid 10001 mlflow \
    && useradd --uid 10001 --gid mlflow --shell /bin/bash --create-home mlflow

# Install MLflow with PostgreSQL and GCS support
RUN pip install --no-cache-dir \
    mlflow==1.30.0 \
    psycopg2-binary \
    google-cloud-storage

# Copy Python entrypoint
COPY start_mlflow.py /home/mlflow/start_mlflow.py

# Set ownership (optional, if start_mlflow.py isn't readable by UID 10001)
RUN chown mlflow:mlflow /home/mlflow/start_mlflow.py

# Switch to unprivileged user (K8s best practice)
USER 10001

# Expose MLflow UI port
EXPOSE 8080

# Run the tracking server using Python script
CMD ["python", "/home/mlflow/start_mlflow.py"]
