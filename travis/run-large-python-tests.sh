#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

# NB: Also add --ignore'd tests to run-small-python-tests.sh
pytest tests --large --ignore=tests/examples --ignore=tests/h2o --ignore=tests/keras \
  --ignore=tests/pytorch --ignore=tests/pyfunc --ignore=tests/sagemaker --ignore=tests/sklearn \
  --ignore=tests/spark --ignore=tests/tensorflow --ignore=tests/azureml --ignore=tests/onnx \
  --ignore=tests/keras_autolog --ignore=tests/tensorflow_autolog --ignore=tests/gluon \
  --ignore=tests/gluon_autolog --ignore=tests/xgboost --ignore=tests/lightgbm \
  --ignore=tests/fastai --ignore=tests/spacy --ignore=tests/spark_autologging --ignore=tests/models

test $err = 0
