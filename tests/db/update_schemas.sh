#!/usr/bin/env bash

set -ex

docker-compose -f tests/db/docker-compose.yml down --volumes --remove-orphans
DEPENDENCIES="$(python setup.py --quiet dependencies)"
docker-compose -f tests/db/docker-compose.yml build --build-arg DEPENDENCIES="$DEPENDENCIES"
docker-compose -f tests/db/docker-compose.yml run --rm mlflow-sqlite python tests/db/test_schema.py
docker-compose -f tests/db/docker-compose.yml run --rm mlflow-postgresql python tests/db/test_schema.py
docker-compose -f tests/db/docker-compose.yml run --rm mlflow-mysql python tests/db/test_schema.py
docker-compose -f tests/db/docker-compose.yml run --rm mlflow-mssql python tests/db/test_schema.py
