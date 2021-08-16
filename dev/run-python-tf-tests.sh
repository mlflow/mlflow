#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

# Run Spark autologging tests, which rely on tensorflow
./dev/test-spark-autologging.sh

test $err = 0
