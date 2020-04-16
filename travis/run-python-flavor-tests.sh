#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

# Run ML framework tests in their own Python processes to avoid OOM issues due to per-framework
# overhead
pytest --color=yes --verbose tests/pytorch --large
pytest --color=yes --verbose tests/h2o --large
pytest --color=yes --verbose tests/onnx --large
pytest --color=yes --verbose tests/pyfunc --large
pytest --color=yes --verbose tests/sklearn --large
pytest --color=yes --verbose tests/azureml --large
pytest --color=yes --verbose tests/models --large
pytest --color=yes --verbose tests/xgboost --large
pytest --color=yes --verbose tests/lightgbm --large
pytest --color=yes --verbose tests/gluon --large
pytest --color=yes --verbose tests/gluon_autolog --large
pytest --color=yes --verbose tests/spacy --large

test $err = 0
