#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

SAGEMAKER_OUT=$(mktemp)
if mlflow sagemaker build-and-push-container --no-push --mlflow-home . > $SAGEMAKER_OUT 2>&1; then
  echo "Sagemaker container build succeeded.";
  # output the last few lines for the timing information (defaults to 10 lines)
else
  echo "Sagemaker container build failed, output:";
  cat $SAGEMAKER_OUT;
fi

ignore=(
  h2o
  pytorch
  pyfunc
  sagemaker
  sklearn
  spark
  azureml
  onnx
  xgboost
  lightgbm
  tensorflow/test_tensorflow_model_export.py
  tensorflow_autolog/test_tensorflow_autolog.py
)

ignore_tf2=(
  tensorflow/test_tensorflow2_model_export.py
  tensorflow_autolog/test_tensorflow2_autolog.py
  keras
  keras_autolog
  gluon
  gluon_autolog
)

ignore_all=("${ignore[@]}" "${ignore_tf2[@]}")

# NB: Also add --ignore'd tests to run-small-python-tests.sh
pytest tests --large "${ignore_all[@]/#/--ignore=tests/}" --ignore tests/spark_autologging

# Run ML framework tests in their own Python processes to avoid OOM issues due to per-framework
# overhead
for path in "${ignore[@]}"
do
  pytest --verbose --large tests/$path
done

# TODO(smurching) Unpin TensorFlow dependency version once test failures with TF 2.1.0 have been
# fixed
pip install 'tensorflow==2.0.0'

for path in "${ignore_tf2[@]}"
do
  pytest --verbose --large tests/$path
done

# Run Spark autologging tests
./travis/test-spark-autologging.sh

test $err = 0
