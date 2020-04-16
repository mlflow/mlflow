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

pytest --verbose tests/sagemaker --large
pytest --verbose tests/sagemaker/mock --large

# Added here due to dependency on sagemaker
# TODO: split out sagemaker tests and move other spark tests to run-python-flavor-tests.sh
pytest --verbose tests/spark --large

test $err = 0
