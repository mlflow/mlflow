#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

pytest --verbose tests/tensorflow/test_tensorflow_model_export.py --large
pytest --verbose tests/tensorflow_autolog/test_tensorflow_autolog.py --large
# TODO(smurching) Unpin TensorFlow dependency version once test failures with TF 2.1.0 have been
# fixed
pip install 'tensorflow==2.0.0'
pytest --verbose tests/tensorflow/test_tensorflow2_model_export.py --large
pytest --verbose tests/tensorflow_autolog/test_tensorflow2_autolog.py --large
pytest --verbose tests/keras --large
pytest --verbose tests/keras_autolog --large

# Run Spark autologging tests, which rely on tensorflow
./travis/test-spark-autologging.sh

test $err = 0
