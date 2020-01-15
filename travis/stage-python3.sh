#!/bin/bash
set -ex
if ! [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$TRAVIS_BUILD_STAGE_NAME" == "Nightly" ]]
then
  if [[ "$TRAVIS_OS_NAME" == "windows" ]]
  then
     echo "skipping this step on windows."
  elif [[ "$TRAVIS_BUILD_STAGE_NAME" == "Small" ]]
  then
    ./travis/run-small-python-tests.sh && ./test-generate-protos.sh
  else
    ./travis/run-large-python-tests.sh
    ./travis/test-anaconda-compatibility.sh
  fi
fi
CHANGED_FILES=$(git diff --name-only master..HEAD | grep "tests/examples\|examples") || true
if [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$CHANGED_FILES" == *"examples"* ]] && [[ "$TRAVIS_BUILD_STAGE_NAME" == "Nightly" ]]
then
  pytest --verbose tests/examples --large;
fi
if [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$CHANGED_FILES" == *"Dockerfile"* ]] && [[ "$TRAVIS_BUILD_STAGE_NAME" == "Nightly" ]]
then
  cddocker build -t mlflow_test_build . && docker images | grep mlflow_test_build
fi
