This directory contains a set of files for demonstrating the mlflow artifacts service.

## Build a wheel

```sh
# Use the development version of mlflow
pip wheel --no-deps --wheel-dir dist ../..

# Use mlflow on PyPI
pip wheel --no-deps --wheel-dir dist mlflow
```

## Run the MLflow Tracking & Artifacts services

```sh
docker-compose up
```

## Log artifacts

```sh
python log_artifacts.py
```

## View the logging results:

- MLflow UI: http://localhost:5000
- MinIO Console: http://localhost:9001 (username: user, password: password)

## Clean up the services

```
docker-compose down --volumes
```
