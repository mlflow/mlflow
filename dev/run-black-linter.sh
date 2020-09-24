#!/usr/bin/env bash

# Exclude proto files because they are auto-generated
black --check --line-length=100 --exclude=mlflow/protos .
exit_code="$?"
if [[ "$exit_code" != "0" ]]; then
  echo "Python lint failed. Run 'black --line-length=100 --exclude=mlflow/protos .' from your"\
    "checkout of MLflow (note the trailing '.' at the end of the command, corresponding to the"\
    "current directory) to autoformat Python code"
  exit $exit_code
fi;
