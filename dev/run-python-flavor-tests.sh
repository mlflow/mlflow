#!/usr/bin/env bash
set -x

pytest \
  tests/utils/test_model_utils.py \
  tests/tracking/fluent/test_fluent_autolog.py \
  tests/autologging \
  tests/server/auth
