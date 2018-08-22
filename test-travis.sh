#!/usr/bin/env bash

set -e
set -x

# Reactivate environment created in the travis installation step
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
source activate test-environment

sudo ./test-generate-protos.sh
pip list
which mlflow
echo $MLFLOW_HOME
mlflow sagemaker build-and-push-container --no-push --mlflow-home .
./lint.sh

# Run tests that don't leverage specific ML frameworks
pytest --cov=mlflow --verbose --large --ignore=tests/h2o --ignore=tests/keras
    --ignore=tests/pytorch --ignore=tests/pyfunc--ignore=tests/sagemaker --ignore=tests/sklearn
    --ignore=tests/spark --ignore=tests/tensorflow

# Run ML framework tests in their own Python processes. TODO: find a better method of isolating
# tests.
pytest --cov=mlflow --verbose tests/h2o --large
pytest --cov=mlflow --verbose tests/keras --large
pytest --cov=mlflow --verbose tests/pytorch --large
pytest --cov=mlflow --verbose tests/pyfunc --large
pytest --cov=mlflow --verbose tests/sagemaker --large
pytest --cov=mlflow --verbose tests/sklearn --large
pytest --cov=mlflow --verbose tests/spark --large
pytest --cov=mlflow --verbose tests/tensorflow --large
cd mlflow/server/js
npm i
./lint.sh
npm test -- --coverage
cd ../../java
mvn clean test
cd ../..
codecov -e TOXENV
