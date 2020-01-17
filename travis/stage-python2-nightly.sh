#!/bin/bash
set -ex
if [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$CHANGED_FILES" == *"examples"* ]]
then
    pytest --verbose tests/examples --large
fi
if [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$CHANGED_FILES" == *"Dockerfile"* ]]
then
    docker build -t mlflow_test_build . && docker images | grep mlflow_test_build
fi
