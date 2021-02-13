#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

# Run ML framework tests in their own Python processes to avoid OOM issues due to per-framework
# overhead
pytest --verbose tests/pytorch --large
pytest --verbose tests/h2o --large
pytest --verbose tests/onnx --large
pytest --verbose tests/pyfunc --large
pytest --verbose tests/sklearn --large
pytest --verbose tests/azureml --large
pytest --verbose tests/models --large
pytest --verbose tests/xgboost --large
pytest --verbose tests/lightgbm --large
pytest --verbose tests/statsmodels --large
pytest --verbose tests/gluon --large
pytest --verbose tests/gluon_autolog --large
pytest --verbose tests/spacy --large
pytest --verbose tests/fastai --large
pytest --verbose tests/shap --large
pytest --verbose tests/utils/test_model_utils.py --large
pytest --verbose tests/tracking/fluent/test_fluent_autolog.py --large


test $err = 0
