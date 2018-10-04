# Docker Image of MLFLow Tracking Server

This subfolder provides a docker image of [MLFLow Tracking Server](https://www.mlflow.org/docs/latest/tracking.html) based on a simple file system (=storage model) for [files and artifacts](https://www.mlflow.org/docs/latest/tracking.html#storage).

## Run

```bash
$ docker build -t mlflow-tracking-server .
```

```bash
$ docker run \
    --rm \
    --name mlflow-tracking-server \
    -p 5000:5000 \
    -e PORT=5000 \
    -e FILE_DIR=/mlflow \
    mlflow-tracking-server
```

Access to http://127.0.0.1:5000

## Environment variables

### Required

|Key|Description|
|---|---|
|`FILE_DIR`|Directory for artifacts and metadata (e.g. parameters, metrics)|

### Optional

|Key|Description|Default|
|---|---|---|
|`PORT`|Value for `listen` directive|`5000`|
