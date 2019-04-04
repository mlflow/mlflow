#!/usr/bin/env bash

pytest --cov=mlflow --verbose --large --ignore=tests/h2o --ignore=tests/keras \
  --ignore=tests/pytorch --ignore=tests/pyfunc --ignore=tests/sagemaker --ignore=tests/sklearn \
  --ignore=tests/spark --ignore=tests/tensorflow

# Run ML framework tests in their own Python processes. TODO: find a better method of isolating
# tests.
pytest --verbose tests/h2o --large
# TODO(smurching): Re-enable Keras tests once they're no longer flaky
# pytest --verbose tests/keras --large
pytest --verbose tests/pytorch --large
pytest --verbose tests/pyfunc --large
pytest --verbose tests/sagemaker --large
pytest --verbose tests/sagemaker/mock --large
pytest --verbose tests/sklearn --large
pytest --verbose tests/spark --large
pytest --verbose tests/tensorflow --large
