#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

# Include testmon database file, assuming it exists
if [ -e "testmon/.testmondata" ]; then
    cp testmon/.testmondata .testmondata
fi

# NB: Also add --ignore'd tests to run-large-python-tests.sh
pytest tests --testmon --suppress-no-test-exit-code --ignore=tests/examples --ignore=tests/h2o --ignore=tests/keras \
  --ignore=tests/pytorch --ignore=tests/pyfunc --ignore=tests/sagemaker --ignore=tests/sklearn \
  --ignore=tests/spark --ignore=tests/tensorflow --ignore=tests/azureml --ignore=tests/onnx \
  --ignore=tests/keras_autolog --ignore=tests/tensorflow_autolog --ignore=tests/gluon \
  --ignore=tests/gluon_autolog --ignore=tests/xgboost --ignore=tests/lightgbm \
  --ignore tests/spark_autologging --ignore=tests/models


# Copy testmon DB file into cache directory. TODO: allow people to run this locally without
# copying into cache directory
mkdir -p testmon
mv .testmondata testmon

test $err = 0
