#!/usr/bin/env bash
set -ex

export MLFLOW_HOME=$(pwd)

pytest tests/sagemaker
