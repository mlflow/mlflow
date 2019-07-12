#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR

# NB: Also add --ignore'd tests to run-large-python-tests.sh
pytest --cov=mlflow --verbose --ignore=tests/h2o --ignore=tests/keras \
  --ignore=tests/pytorch --ignore=tests/pyfunc --ignore=tests/sagemaker --ignore=tests/sklearn \
  --ignore=tests/spark --ignore=tests/tensorflow --ignore=tests/autologging \
  --ignore tests/azureml --ignore tests/onnx tests

test $err = 0
