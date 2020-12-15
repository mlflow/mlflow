#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

# NB: Also add --ignore'd tests to run-small-python-tests.sh
pytest tests --large --ignore-flavors --ignore=tests/examples

test $err = 0
