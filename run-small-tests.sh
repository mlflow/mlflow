#!/usr/bin/env bash

pytest --cov=mlflow --verbose --large --ignore=tests/h2o --ignore=tests/keras \
  --ignore=tests/pytorch --ignore=tests/pyfunc --ignore=tests/sagemaker --ignore=tests/sklearn \
  --ignore=tests/spark --ignore=tests/tensorflow

