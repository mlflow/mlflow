# mlflow docker image

## Image variants

### mlflow:\<version\>

This image contains mlflow plus all extra dependencies. It is intended to support "out of the box" all mlflow integrations.

Use this image if you just want to quickly have a working instance of mlflow, or if you don't want to bother rolling out your own image.

### mlflow:\<version\>-slim

This image only contains the core mlflow dependencies, so most of the integrations (ie. backend stores databases,
artifact stores, etc.) will not work.

Use this image if you don't need any integration, or as the base image in case you want to have full control over
the extra dependencies installed.

# Testing

The [testcontainers](https://testcontainers.com/) package is used to the test the different integrations, for each one there
is a Docker Compose file. So far the following integrations are tested:

Backend store:

- MySQL
- PostgreSQL
- MSSQL

Artifact store:

- Amazon S3 (through the [minio](https://hub.docker.com/r/minio/minio) image)
- Azure Blob Storage (throug the [Azurite](https://hub.docker.com/_/microsoft-azure-storage-azurite) image)
- Google Cloud Storage (through the [fake-gcs-server](https://github.com/fsouza/fake-gcs-server) image)

## Executing the tests

Make sure that you install all required dependencies first.

```bash
# mlflow, from a release
$ pip install -U mlflow
# or in editable mode
$ pip install -e .
# testcontaienrs & pytest
$ pip install -U testcontainers<4 pytest
```

Ensure [Docker](https://www.docker.com/) is installed. You will also need `docker-compose`, make sure to install
[docker-compose v1](https://docs.docker.com/compose/install/standalone/), since `testcontainers` only supports Compose v2
from version 4.0 onwards, which is available only for Python 3.9+.

Once you have all the dependencies you can execute the tests (from within `docker` folder):

```bash
# Execute all tests
pytest
# Execute a specific test
pytest -k 'test_backend_and_artifact_store_integration[docker-compose.aws-test.yaml]'
```
