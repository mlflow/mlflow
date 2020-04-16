#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

# NB: Also add --ignore'd tests to run-small-python-tests.sh
pytest tests --color=yes --large --ignore=tests/examples --ignore=tests/h2o --ignore=tests/keras \
  --ignore=tests/pytorch --ignore=tests/pyfunc --ignore=tests/sagemaker --ignore=tests/sklearn \
  --ignore=tests/spark --ignore=tests/tensorflow --ignore=tests/azureml --ignore=tests/onnx \
  --ignore=tests/keras_autolog --ignore=tests/tensorflow_autolog --ignore=tests/gluon \
  --ignore=tests/gluon_autolog --ignore=tests/xgboost --ignore=tests/lightgbm \
  --ignore=tests/spacy --ignore=tests/spark_autologging --ignore=tests/models
# Run ML framework tests in their own Python processes to avoid OOM issues due to per-framework
# overhead
pytest --verbose tests/pytorch --large
pytest --verbose tests/h2o --large
pytest --verbose tests/onnx --large
pytest --verbose tests/pyfunc --large
pytest --verbose tests/sklearn --large
pytest --verbose tests/spark --large
pytest --verbose tests/azureml --large
pytest --verbose tests/models --large
pytest --verbose tests/xgboost --large
pytest --verbose tests/lightgbm --large

test $err = 0
