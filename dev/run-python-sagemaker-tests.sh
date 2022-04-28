#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

for env_manager in conda virtualenv; do
  SAGEMAKER_OUT=$(mktemp)
  if mlflow sagemaker build-and-push-container --no-push --mlflow-home . --env-manager $env_manager > $SAGEMAKER_OUT 2>&1; then
    echo "Sagemaker container build with $env_manager succeeded.";
    # output the last few lines for the timing information (defaults to 10 lines)
  else
    echo "Sagemaker container build with $env_manager failed, output:";
    cat $SAGEMAKER_OUT;
  fi
done

pytest tests/sagemaker --large
pytest tests/sagemaker/mock --large

test $err = 0
