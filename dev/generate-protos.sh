#!/usr/bin/env bash

set -ex
PROTOC_VERSION="$(protoc --version)"
if [[ "$PROTOC_VERSION" != 'libprotoc 3.19.4' ]]; then
	echo "Required libprotoc versions to be 3.19.4 (preferred)."
	echo "We found: $PROTOC_VERSION"
	exit 1
fi
PROTOS="mlflow/protos"
protoc \
    --python_out=. \
    --java_out="mlflow/java/client/src/main/java" \
    "$PROTOS"/databricks.proto \
    "$PROTOS"/service.proto \
    "$PROTOS"/model_registry.proto \
    "$PROTOS"/databricks_artifacts.proto \
    "$PROTOS"/mlflow_artifacts.proto \
    "$PROTOS"/internal.proto \
    "$PROTOS"/scalapb/scalapb.proto \

# Separate out building UC model registry protos to avoid autogenerating
# Java stubs, for now
protoc \
    --python_out=. \
    "$PROTOS"/databricks_uc_registry_messages.proto \
    "$PROTOS"/databricks_uc_registry_service.proto


PROTOS="mlflow/protos"
protoc \
    --python_out=. \
    "$PROTOS"/facet_feature_statistics.proto \

# Generate only Python classes (no Java classes) for test protos.
TEST_PROTOS="tests/protos"
protoc \
    --python_out=. \
    "$TEST_PROTOS"/test_message.proto
