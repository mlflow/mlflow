This directory contains a set of files for demonstrating the MLflow Artifacts Service.

## Quick start

```sh
# Launch tracking and artifacts servers
mlflow server --default-artifact-root http://localhost:5000/api/2.0/mlflow-artifacts/artifacts

# Run `run.py` that performs upload/download/list operations for artifacts
MLFLOW_TRACKING_URI=http://localhost:5000 python run.py
```

### Clean up

```sh
# Remove experiment and run data
rm -rf mlruns

# Remove artifacts
rm -rf mlartifacts
```

MLflow UI is available at http://localhost:5000 to explore the logging results.

## Advanced example using `docker-compose`

```sh
# Build services
docker-compose build

# Launch tracking and artifacts servers in the background
docker-compose up -d

# Run `run.py` in the client container
docker-compose run -v ${PWD}/run.py:/app/run.py client python run.py
```

- MLflow UI is available at http://localhost:5000 to explore the logging results.
- MinIO Console is available at http://localhost:9001 to explore the logged artifacts. The login username and password are `user` and `password`.

### Clean up

```sh
# Remove containers, networks, volumes, and images
docker-compose --rmi all --volumes --remove-orphans
```

### Development

```sh
# Build services using the dev version of mlflow
./build.sh
docker-compose run -v ${PWD}/run.py:/app/run.py client python run.py
```
