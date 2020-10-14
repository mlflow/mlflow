#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

# Run ML framework tests in their own Python processes to avoid OOM issues due to per-framework
# overhead
test_dirs=(
  tests/pytorch
  tests/h2o
  tests/onnx
  tests/pyfunc
  tests/sklearn
  tests/azureml
  tests/models
  tests/xgboost
  tests/lightgbm
  tests/gluon
  tests/gluon_autolog
  tests/spacy
  tests/fastai
  tests/utils/test_model_utils.py
)

for dir in "${test_dirs[@]}"; do
  find $dir -name 'test*.py' | xargs -L 1 pytest --verbose --large
done

test $err = 0
