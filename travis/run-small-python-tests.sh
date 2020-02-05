#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

# Include testmon database file
cp testmon/.testmondata .testmondata

# NB: Also add --ignore'd tests to run-large-python-tests.sh
pytest tests --testmon --ignore=tests/h2o --ignore=tests/keras \
  --ignore=tests/pytorch --ignore=tests/pyfunc --ignore=tests/sagemaker --ignore=tests/sklearn \
  --ignore=tests/spark --ignore=tests/tensorflow --ignore=tests/keras_autolog \
  --ignore=tests/tensorflow_autolog --ignore tests/azureml --ignore tests/onnx \
  --ignore=tests/xgboost --ignore=tests/lightgbm tests --ignore=tests/spark_autologging --ignore=tests/models --ignore=tests/examples

# We expect this to not run the same tests again/exit immediately
pytest tests --testmon --ignore=tests/h2o --ignore=tests/keras \
  --ignore=tests/pytorch --ignore=tests/pyfunc --ignore=tests/sagemaker --ignore=tests/sklearn \
  --ignore=tests/spark --ignore=tests/tensorflow --ignore=tests/keras_autolog \
  --ignore=tests/tensorflow_autolog --ignore tests/azureml --ignore tests/onnx \
  --ignore=tests/xgboost --ignore=tests/lightgbm tests --ignore=tests/spark_autologging --ignore=tests/models --ignore=tests/examples

test $err = 0

# Copy testmon DB file into cache directory. TODO: allow people to run this locally without
# copying into cache directory
mv .testmondata testmon
