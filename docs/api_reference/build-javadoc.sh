#!/usr/bin/env bash

# Builds the MLflow Javadoc and places it into build/html/java_api/

set -ex
pushd ../../mlflow/java/client/
# the MAVEN_JAVADOC_ARGS env var is used to dynamically pass
# args to the mvn command. this can be used to direct maven to use
# a mirror, in case we encounter rate limiting from maven central
mvn clean javadoc:javadoc ${MAVEN_JAVADOC_ARGS} -q
popd
rm -rf build/html/java_api/
mkdir -p build/html/java_api/
cp -r ../../mlflow/java/client/target/site/apidocs/* build/html/java_api/
echo "Copied JavaDoc into docs/build/html/java_api/"
