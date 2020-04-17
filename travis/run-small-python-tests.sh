#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

# NB: Also add --ignore'd tests to run-large-python-tests.sh
pytest --cov=mlflow --verbose --ignore=tests/h2o --ignore=tests/keras \
  --ignore=tests/pytorch --ignore=tests/pyfunc --ignore=tests/sagemaker --ignore=tests/sklearn \
  --ignore=tests/spark --ignore=tests/tensorflow --ignore=tests/keras_autolog \
  --ignore=tests/tensorflow_autolog --ignore tests/azureml --ignore tests/onnx \
  --ignore=tests/xgboost --ignore=tests/spacy --ignore=tests/lightgbm tests --ignore=tests/spark_autologging

test $err = 0
