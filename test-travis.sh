#!/usr/bin/env bash

set -e
set -x

source activate test-environment

sudo ./test-generate-protos.sh
pip list
which mlflow
echo $MLFLOW_HOME
mlflow sagemaker build-and-push-container --no-push --mlflow-home .
pytest --cov=mlflow --verbose --large
./lint.sh
cd mlflow/server/js
npm i
npm test -- --coverage
cd ../../..
codecov -e TOXENV
