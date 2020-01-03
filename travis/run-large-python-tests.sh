#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

SAGEMAKER_OUT=$(mktemp)
if time mlflow sagemaker build-and-push-container --no-push --mlflow-home . > $SAGEMAKER_OUT 2>&1; then
  echo "Sagemaker container build succeeded.";
  cat $SAGEMAKER_OUT;
else
  echo "Sagemaker container build failed, output:";
  cat $SAGEMAKER_OUT;
fi

# NB: Also add --ignore'd tests to run-small-python-tests.sh
pytest tests --large --ignore=tests/examples --ignore=tests/h2o --ignore=tests/keras \
  --ignore=tests/pytorch --ignore=tests/pyfunc --ignore=tests/sagemaker --ignore=tests/sklearn \
  --ignore=tests/spark --ignore=tests/tensorflow --ignore=tests/azureml --ignore=tests/onnx \
  --ignore=tests/keras_autolog --ignore=tests/tensorflow_autolog --ignore=tests/gluon \
  --ignore=tests/gluon_autolog --ignore=tests/xgboost --ignore=tests/lightgbm
# Run ML framework tests in their own Python processes to avoid OOM issues due to per-framework
# overhead
time pytest --verbose tests/h2o --large
time pytest --verbose tests/onnx --large
time pytest --verbose tests/pytorch --large
time pytest --verbose tests/pyfunc --large
time pytest --verbose tests/sagemaker --large
time pytest --verbose tests/sagemaker/mock --large
time pytest --verbose tests/sklearn --large
time pytest --verbose tests/spark --large
time pytest --verbose tests/tensorflow/test_tensorflow_model_export.py --large
time pytest --verbose tests/tensorflow_autolog/test_tensorflow_autolog.py --large
time pytest --verbose tests/azureml --large
time pytest --verbose tests/models --large
time pytest --verbose tests/xgboost --large
time pytest --verbose tests/lightgbm --large
time pip install 'tensorflow>=2.0.0'
time pytest --verbose tests/tensorflow/test_tensorflow2_model_export.py --large
time pytest --verbose tests/tensorflow_autolog/test_tensorflow2_autolog.py --large
time pytest --verbose tests/keras --large
time pytest --verbose tests/keras_autolog --large
time pytest --verbose tests/gluon --large
time pytest --verbose tests/gluon_autolog --large
test $err = 0
