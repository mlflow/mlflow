#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

# Run ML framework tests in their own Python processes to avoid OOM issues due to per-framework
# overhead
pytest tests/azureml
pytest tests/utils/test_model_utils.py

# TODO: Run tests for h2o, shap, and paddle in the cross-version-tests workflow
pytest tests/h2o
pytest tests/shap
pytest tests/paddle

pytest tests/tracking/fluent/test_fluent_autolog.py
pytest tests/autologging
pytest tests/test_mlflow_lazily_imports_ml_packages.py --lazy-import

test $err = 0
