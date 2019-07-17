#!/usr/bin/env bash

# Builds the MLflow Javadoc and places it into build/html/java_api/

set -ex
pushd ../mlflow/java/client/
mvn clean javadoc:javadoc
popd
rm -rf build/html/java_api/
mkdir -p build/html/java_api/
cp -r ../mlflow/java/client/target/site/apidocs/* build/html/java_api/
echo "Copied JavaDoc into docs/build/html/java_api/"
