# Genesis-Flow Release Guide

This document describes how to release and publish Genesis-Flow as a Python package.

## Release Process

### 1. Version Management

The version is managed in `pyproject.toml`:
```toml
[project]
name = "genesis-flow"
version = "1.0.0"
```

### 2. Publishing to PyPI

To publish to PyPI:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Upload to PyPI (requires credentials)
python -m twine upload dist/*
```

### 3. Publishing to Private Repository

For private repositories (e.g., Azure Artifacts, Artifactory):

```bash
# Configure your private repository
pip config set global.index-url https://your-private-repo/simple/

# Or use .pypirc file
# ~/.pypirc
[distutils]
index-servers =
    pypi
    private

[private]
repository = https://your-private-repo/
username = your-username
password = your-token

# Upload to private repo
python -m twine upload -r private dist/*
```

### 4. Installing from Git

For development or before publishing:

```bash
# Install directly from git
pip install git+https://github.com/autonomize-ai/genesis-flow.git

# Or in pyproject.toml
[tool.poetry.dependencies]
genesis-flow = {git = "https://github.com/autonomize-ai/genesis-flow.git", branch = "main"}

# Or in requirements.txt
git+https://github.com/autonomize-ai/genesis-flow.git@main#egg=genesis-flow
```

### 5. Installing from Private Repository

Once published:

```bash
# Install from private repo
pip install genesis-flow --index-url https://your-private-repo/simple/

# Or in pyproject.toml
[tool.poetry.dependencies]
genesis-flow = "^1.0.0"

[tool.poetry.source]
name = "private"
url = "https://your-private-repo/simple/"
priority = "primary"
```

## Key Features Added

### 1. MLFLOW_ARTIFACT_LOCATION Support

Genesis-Flow now supports the `MLFLOW_ARTIFACT_LOCATION` environment variable to set a default artifact location for all experiments:

```python
# Set in environment
export MLFLOW_ARTIFACT_LOCATION="gs://my-bucket/mlflow-artifacts"

# Or in code
import os
os.environ["MLFLOW_ARTIFACT_LOCATION"] = "./mlflow-artifacts"

# All new experiments will use this location by default
client.create_experiment("my-experiment")  # Uses MLFLOW_ARTIFACT_LOCATION
```

### 2. PostgreSQL with Managed Identity

Full support for Azure PostgreSQL with Managed Identity authentication:

```python
# Connection string with managed identity
MLFLOW_TRACKING_URI="postgresql://user@server.postgres.database.azure.com:5432/mlflow?auth_method=managed_identity"
```

### 3. Google Cloud Storage Support

Native support for GCS as artifact store:

```python
# Use GCS for artifacts
MLFLOW_ARTIFACT_LOCATION="gs://my-bucket/mlflow-artifacts"
```

## Migration from MLflow

Genesis-Flow is a drop-in replacement for MLflow:

```python
# Before
import mlflow
from mlflow.tracking import MlflowClient

# After
import mlflow  # genesis-flow provides the same module
from mlflow.tracking import MlflowClient

# No code changes needed!
```

## Version Compatibility

- Python: >= 3.9
- Compatible with MLflow APIs
- Backward compatible with existing MLflow tracking stores