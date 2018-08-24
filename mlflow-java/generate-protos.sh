#!/usr/bin/env bash

set -e
PROTOC_VERSION="$(protoc --version)"
if [ "$PROTOC_VERSION" != "libprotoc 3.6.0" ]; then
    echo "Must have libprotoc version 3.6.0."
    echo "We found: $PROTOC_VERSION"
    exit 1
fi

protoc -I=../mlflow/protos --java_out=src/main/java service.proto databricks.proto
