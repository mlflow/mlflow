#!/bin/bash
set -ex

uv run tests/store/dump_schema.py tests/resources/db/latest_schema.sql

./tests/db/compose.sh down --volumes --remove-orphans
# TODO: Remove DOCKER_BUILDKIT=0 once https://github.com/moby/buildkit/pull/6477 is available in the runner image
DOCKER_BUILDKIT=0 ./tests/db/compose.sh build --build-arg DEPENDENCIES="$(uv run dev/extract_deps.py)"
for service in $(./tests/db/compose.sh config --services | grep '^mlflow-')
do
  ./tests/db/compose.sh run --rm $service python tests/db/test_schema.py
done
