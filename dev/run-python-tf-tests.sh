#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

pytest tests/tensorflow/test_tensorflow2_model_export.py --large
pytest tests/tensorflow_autolog/test_tensorflow2_autolog.py --large
pytest tests/keras --large
pytest tests/keras_autolog --large
# Downgrade TensorFlow and Keras in order to test compatibility with older versions
pip install 'tensorflow==1.15.4'
pip install 'keras==2.2.5'
pytest tests/keras --large
pytest tests/tensorflow/test_tensorflow_model_export.py --large
pytest tests/tensorflow_autolog/test_tensorflow_autolog.py --large
pytest tests/keras_autolog --large

# Run Spark autologging tests, which rely on tensorflow
./dev/test-spark-autologging.sh

test $err = 0
