#!/usr/bin/env bash

set -ex

./build_wheel.sh
docker-compose down --volumes --remove-orphans
docker-compose pull
docker image ls | grep -E '(REPOSITORY|postgres|mysql|mssql)'
docker-compose build
docker-compose run mlflow-sqlite
docker-compose run mlflow-postgres
docker-compose run mlflow-mysql
docker-compose run mlflow-mssql
