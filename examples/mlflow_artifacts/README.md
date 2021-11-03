This directory contains a set of files for demonstrating the MLflow Artifacts service.

## Build a wheel

This example requires a wheel for `mlflow` to be stored in `dist`.

```sh
# Clean up existing wheels
rm dist/*

# Build a wheel for the development version of mlflow
pip wheel --no-deps --wheel-dir dist ../..

# Build a wheel for the latest version of mlflow on PyPI
pip wheel --no-deps --wheel-dir dist mlflow
```

## Run the example

```sh
# Build services
docker-compose build

# Launch tracking and artifacts servers
docker-compose up -d

# Run `run.py` that uploads, downloads, and list artifacts
docker-compose run -v ${PWD}/run.py:/app/run.py client python run.py
```

## Explore the logging results:

```sh
# Make sure both tracking and artifacts servers are running
docker-compose ps
```

- MLflow UI is available at http://localhost:5000 to explore tracking results.
- MinIO Console is available at http://localhost:9001 to explore logged artifacts. The login username and password are `user` and `password`.

## Reset tracking and artifacts servers

```
docker-compose down --volumes --remove-orphans
```
