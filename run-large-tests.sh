#!/usr/bin/env bash
set -x

# TODO(czumar): Re-enable container building and associated SageMaker tests once the container
# build process is no longer hanging
# - SAGEMAKER_OUT=$(mktemp)
# - if mlflow sagemaker build-and-push-container --no-push --mlflow-home . > $SAGEMAKER_OUT 2>&1; then
#     echo "Sagemaker container build succeeded.";
#   else
#     echo "Sagemaker container build failed, output:";
#     cat $SAGEMAKER_OUT;
#   fi
pytest --cov=mlflow --verbose --large-only --ignore=tests/h2o --ignore=tests/keras \
  --ignore=tests/pytorch --ignore=tests/pyfunc --ignore=tests/sagemaker --ignore=tests/sklearn \
  --ignore=tests/spark --ignore=tests/tensorflow
# Run ML framework tests in their own Python processes to avoid OOM issues due to per-framework
# overhead
pytest --verbose tests/h2o --large
# TODO(smurching): Re-enable Keras tests once they're no longer flaky
# - pytest --verbose tests/keras --large
pytest --verbose tests/pytorch --large
pytest --verbose tests/pyfunc --large
pytest --verbose tests/sagemaker --large
pytest --verbose tests/sagemaker/mock --large
pytest --verbose tests/sklearn --large
pytest --verbose tests/spark --large
pytest --verbose tests/tensorflow --large
