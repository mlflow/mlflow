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

# Include testmon database file, assuming it exists
#if [ -e "testmon/.testmondata" ]; then
#    cp testmon/.testmondata .testmondata
#fi

# Run ML framework tests in their own Python processes to avoid OOM issues due to per-framework
# overhead
pytest --testmon tests/pytorch
pytest --testmon tests/h2o
pytest --testmon tests/onnx
pytest --testmon tests/sagemaker
pytest --testmon tests/sagemaker/mock
pytest --testmon tests/sklearn
pytest --testmon tests/tensorflow/test_tensorflow_model_export.py
pytest --testmon tests/tensorflow_autolog/test_tensorflow_autolog.py
pytest --testmon tests/azureml
pytest --testmon tests/models
pytest --testmon tests/xgboost
pytest --testmon tests/lightgbm
# TODO(smurching) Unpin TensorFlow dependency version once test failures with TF 2.1.0 have been
# fixed
pip install 'tensorflow==2.0.0'
pytest --testmon tests/tensorflow/test_tensorflow2_model_export.py
pytest --testmon tests/tensorflow_autolog/test_tensorflow2_autolog.py
pytest --testmon tests/keras
pytest --testmon tests/keras_autolog
pytest --testmon tests/gluon
pytest --testmon tests/gluon_autolog
pytest --testmon tests/pyfunc --capture=no
pytest --testmon tests/spark --capture=no

# Run Spark autologging tests
./travis/test-spark-autologging.sh

# Copy testmon DB file into cache directory. TODO: allow people to run this locally without
# copying into cache directory
mkdir -p testmon
mv .testmondata testmon

test $err = 0
