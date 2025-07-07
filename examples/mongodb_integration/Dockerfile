FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY examples/mongodb_integration/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Genesis-Flow source code
COPY . /app/

# Install Genesis-Flow in development mode
RUN pip install -e .

# Create artifacts directory
RUN mkdir -p /tmp/genesis_flow_artifacts

# Set Python path
ENV PYTHONPATH=/app

# Set default environment variables
ENV MLFLOW_TRACKING_URI="mongodb://genesis:genesis_flow_test@mongodb:27017/genesis_flow_test?authSource=admin"
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT="/tmp/genesis_flow_artifacts"

# Default command
CMD ["python", "/app/examples/mongodb_integration/test_genesis_flow_mongodb.py"]