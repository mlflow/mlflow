#!/bin/bash

set -e

start() {
    if [[ $DATABRICKS_CONFIG_WEBPACK_MLFLOW == 'true' ]];
    then
        echo "[start] Using databricks-webpack because DATABRICKS_CONFIG_WEBPACK_MLFLOW=true"
        databricks-webpack start
    else
        craco start
    fi
}

build() {
    if [[ $DATABRICKS_CONFIG_WEBPACK_MLFLOW == 'true' ]];
    then
        echo "[build] Using databricks-webpack because DATABRICKS_CONFIG_WEBPACK_MLFLOW=true"
        databricks-webpack build
    else
        craco --max_old_space_size=8192 build
    fi
}

test() {
    if [[ $DATABRICKS_CONFIG_JEST_MLFLOW == 'true' ]];
    then
        echo "[build] Using databricks-jest because DATABRICKS_CONFIG_JEST_MLFLOW=true"
        databricks-jest "${@:1}"
    else
        craco --max_old_space_size=8192 test --env=jsdom --colors --watchAll=false "${@:1}"
    fi
}

test:watch() {
    if [[ $DATABRICKS_CONFIG_JEST_MLFLOW == 'true' ]];
    then
        echo "[build] Using databricks-jest because DATABRICKS_CONFIG_JEST_MLFLOW=true"
        databricks-jest --watch "${@:1}"
    else
        craco --max_old_space_size=8192 test --env=jsdom --colors --watchAll=false --watch "${@:1}"
    fi
}

test:ci() {
    if [[ $DATABRICKS_CONFIG_JEST_MLFLOW == 'true' ]];
    then
        echo "[build] Using databricks-jest because DATABRICKS_CONFIG_JEST_MLFLOW=true"
        CI=true databricks-jest --forceExit --ci --coverage
    else
        CI=true craco test --env=jsdom --colors --forceExit --ci --coverage
    fi
}

eval "$1" "${@:2}"
