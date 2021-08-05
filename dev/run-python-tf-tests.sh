#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)


# TODO: Fix `test_spark_datasource_autologging_crossframework.py` and remove this line
pip install 'tensorflow==1.15.4' 'keras==2.2.5'

# Run Spark autologging tests, which rely on tensorflow
./dev/test-spark-autologging.sh

test $err = 0
