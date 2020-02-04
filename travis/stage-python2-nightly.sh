#!/bin/bash
set -ex
CHANGED_FILES=$(git diff --name-only master..HEAD | grep "tests/examples\|examples\|Dockerfile") || true
# Set matplotlib to not use the Xwindows backend which causes an error.
echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc
if [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$CHANGED_FILES" == *"examples"* ]]
then
    pytest --verbose tests/examples --large
fi
if [[ "$TRAVIS_EVENT_TYPE" == "cron" || "$CHANGED_FILES" == *"Dockerfile"* ]]
then
    docker build -t mlflow_test_build . && docker images | grep mlflow_test_build
fi
