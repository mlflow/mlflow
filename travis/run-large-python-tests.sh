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
pytest --testmon --suppress-no-test-exit-code tests/pytorch
pytest --testmon --suppress-no-test-exit-code tests/h2o
pytest --testmon --suppress-no-test-exit-code tests/onnx
pytest --testmon --suppress-no-test-exit-code tests/sagemaker
pytest --testmon --suppress-no-test-exit-code tests/sagemaker/mock
pytest --testmon --suppress-no-test-exit-code tests/sklearn
pytest --testmon --suppress-no-test-exit-code tests/tensorflow/test_tensorflow_model_export.py
pytest --testmon --suppress-no-test-exit-code tests/tensorflow_autolog/test_tensorflow_autolog.py
pytest --testmon --suppress-no-test-exit-code tests/azureml
pytest --testmon --suppress-no-test-exit-code tests/models
pytest --testmon --suppress-no-test-exit-code tests/xgboost
pytest --testmon --suppress-no-test-exit-code tests/lightgbm
# TODO(smurching) Unpin TensorFlow dependency version once test failures with TF 2.1.0 have been
# fixed
pip install 'tensorflow==2.0.0'
pytest --testmon --suppress-no-test-exit-code tests/tensorflow/test_tensorflow2_model_export.py
pytest --testmon --suppress-no-test-exit-code tests/tensorflow_autolog/test_tensorflow2_autolog.py
pytest --testmon --suppress-no-test-exit-code tests/keras
pytest --testmon --suppress-no-test-exit-code tests/keras_autolog
pytest --testmon --suppress-no-test-exit-code tests/gluon
pytest --testmon --suppress-no-test-exit-code tests/gluon_autolog
pytest --testmon --suppress-no-test-exit-code tests/pyfunc --capture=no
pytest --testmon --suppress-no-test-exit-code tests/spark --capture=no

# Run Spark autologging tests
./travis/test-spark-autologging.sh

# Copy testmon DB file into cache directory. TODO: allow people to run this locally without
# copying into cache directory
mkdir -p testmon
mv .testmondata testmon

test $err = 0
