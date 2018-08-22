#!/usr/bin/env bash

set -e
set -x

sudo ./test-generate-protos.sh
pip list
which mlflow
echo $MLFLOW_HOME
mlflow sagemaker build-and-push-container --no-push --mlflow-home .
./lint.sh
pytest --cov=mlflow --verbose --large
cd mlflow/server/js
npm i
./lint.sh
npm test -- --coverage
cd ../../java
mvn clean test
cd ../..
codecov -e TOXENV
