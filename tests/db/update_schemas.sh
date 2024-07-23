#!/bin/bash
set -ex

python tests/store/dump_schema.py tests/resources/db/latest_schema.sql

./tests/db/compose.sh down --volumes --remove-orphans
./tests/db/compose.sh build --build-arg DEPENDENCIES="$(cat requirements/skinny-requirements.txt requirements/core-requirements.txt | grep -Ev '^(#|$)')"
for service in $(./tests/db/compose.sh config --services | grep '^mlflow-')
do
  ./tests/db/compose.sh run --rm $service python tests/db/test_schema.py
done
