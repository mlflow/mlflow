#!/usr/bin/env bash
set -e

SKINNY=0
PYTHON_PATH=python

rm -rf build dist mlflow.egg-info mlflow_skinny.egg-info skinny/mlflow skinny/mlflow_skinny.egg-info

for arg in "$@"
do
  if [ "$arg" == "--skinny" ]; then
    SKINNY=1
  elif [[ "$arg" == --python-path=* ]]; then
    PYTHON_PATH="${arg#*=}"
  fi
done

if [ $SKINNY -eq 1 ]; then
  $PYTHON_PATH -m build skinny --outdir dist
else
  $PYTHON_PATH -m build
fi
