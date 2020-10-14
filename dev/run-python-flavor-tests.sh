#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

# Run ML framework tests in their own Python processes to avoid OOM issues due to per-framework
# overhead
paths=(
  # tests/pytorch
  # tests/h2o
  # tests/onnx
  # tests/pyfunc
  tests/sklearn
  # tests/azureml
  # tests/models
  # tests/xgboost
  # tests/lightgbm
  # tests/gluon
  # tests/gluon_autolog
  # tests/spacy
  # tests/fastai
  tests/utils/test_model_utils.py
)

pytest_options="--verbose --large"

for path in "${paths[@]}"; do
  if [ -f "$path" ]; then
    pytest $path $pytest_options
    echo $?
  else
    for path in $(find $path -name 'test*.py'); do
      pytest $path $pytest_options
      echo $?
    done
  fi
done

test $err = 0
