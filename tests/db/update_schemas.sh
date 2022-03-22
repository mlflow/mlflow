#!/bin/bash
set -ex

./tests/db/compose.sh down --volumes --remove-orphans
./tests/db/compose.sh build --build-arg DEPENDENCIES="$(python setup.py -q dependencies)"
for service in $(./tests/db/compose.sh config --services | grep '^mlflow-')
do
  ./tests/db/compose.sh run --rm $service python tests/db/test_schema.py
done
