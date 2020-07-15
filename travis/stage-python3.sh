#!/bin/bash
set -ex
if ! [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$TRAVIS_BUILD_STAGE_NAME" == "Nightly" ]]
then
  ./travis/test-anaconda-compatibility.sh "anaconda3:2020.02"
  ./travis/test-anaconda-compatibility.sh "anaconda3:2019.03"
fi
CHANGED_FILES=$(git diff --name-only master..HEAD | grep "tests/examples\|examples") || true
if [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$CHANGED_FILES" == *"examples"* ]]
then
  pytest --verbose tests/examples --large;
fi
if [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$CHANGED_FILES" == *"Dockerfile"* ]]
then
  docker build -t mlflow_test_build . && docker images | grep mlflow_test_build
fi
