#!/usr/bin/env bash
set -x

export MLFLOW_HOME=$(pwd)

pytest \
  tests/utils/test_model_utils.py \
  tests/tracking/fluent/test_fluent_autolog.py \
  tests/autologging
