#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR


# TODO(czumar): Re-enable container building and associated SageMaker tests once the container
# build process is no longer hanging
# - SAGEMAKER_OUT=$(mktemp)
# - if mlflow sagemaker build-and-push-container --no-push --mlflow-home . > $SAGEMAKER_OUT 2>&1; then
#     echo "Sagemaker container build succeeded.";
#   else
#     echo "Sagemaker container build failed, output:";
#     cat $SAGEMAKER_OUT;
#   fi
# NB: Also add --ignore'd tests to run-small-python-tests.sh
pytest tests --large --ignore=tests/h2o --ignore=tests/keras \
  --ignore=tests/pytorch --ignore=tests/pyfunc --ignore=tests/sagemaker --ignore=tests/sklearn \
  --ignore=tests/spark --ignore=tests/tensorflow --ignore=tests/azureml --ignore=tests/onnx \
  --ignore=tests/autologging
# Run ML framework tests in their own Python processes to avoid OOM issues due to per-framework
# overhead
pytest --verbose tests/h2o --large
# TODO(smurching): Re-enable Keras tests once they're no longer flaky
# - pytest --verbose tests/keras --large
pytest --verbose tests/onnx --large;
pytest --verbose tests/pytorch --large
pytest --verbose tests/pyfunc --large
pytest --verbose tests/sagemaker --large
pytest --verbose tests/sagemaker/mock --large
pytest --verbose tests/sklearn --large
pytest --verbose tests/spark --large
pytest --verbose tests/tensorflow --large
pytest --verbose tests/azureml --large
pytest --verbose tests/models --large
pytest --verbose tests/autologging --large
test $err = 0
