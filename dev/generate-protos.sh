#!/usr/bin/env bash

set -ex
PROTOC_VERSION="$(protoc --version)"
if [[ "$PROTOC_VERSION" != 'libprotoc 3.19.4' ]]; then
	echo "Required libprotoc versions to be 3.19.4 (preferred)."
	echo "We found: $PROTOC_VERSION"
	exit 1
fi
PROTOS="mlflow/protos"
protoc -I="$PROTOS" \
    --python_out="$PROTOS" \
    --java_out="mlflow/java/client/src/main/java" \
    "$PROTOS"/databricks.proto \
    "$PROTOS"/service.proto \
    "$PROTOS"/model_registry.proto \
    "$PROTOS"/databricks_artifacts.proto \
    "$PROTOS"/mlflow_artifacts.proto \
    "$PROTOS"/internal.proto \
    "$PROTOS"/scalapb/scalapb.proto \

# Separate out building UC model registry and managed catalog protos to avoid
# autogenerating Java stubs, for now
protoc -I="$PROTOS" \
    --python_out="$PROTOS" \
    "$PROTOS"/databricks_managed_catalog_messages.proto \
    "$PROTOS"/databricks_managed_catalog_service.proto \
    "$PROTOS"/databricks_uc_registry_messages.proto \
    "$PROTOS"/databricks_uc_registry_service.proto \
    "$PROTOS"/databricks_filesystem_service.proto


PROTOS="mlflow/protos"
protoc -I="$PROTOS" \
    --python_out="$PROTOS" \
    "$PROTOS"/facet_feature_statistics.proto \

# Generate only Python classes (no Java classes) for test protos.
TEST_PROTOS="tests/protos"
protoc -I="$TEST_PROTOS" \
    --python_out="$TEST_PROTOS" \
    "$TEST_PROTOS"/test_message.proto

OLD_SCALAPB="from scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2"
NEW_SCALAPB="from .scalapb import scalapb_pb2 as scalapb_dot_scalapb__pb2"
sed -i'.old' -e "s/$OLD_SCALAPB/$NEW_SCALAPB/g" "$PROTOS/databricks_pb2.py" "$PROTOS/service_pb2.py" "$PROTOS/model_registry_pb2.py" "$PROTOS/databricks_artifacts_pb2.py" "$PROTOS/mlflow_artifacts_pb2.py" "$PROTOS/internal_pb2.py" "$PROTOS/databricks_managed_catalog_service_pb2.py" "$PROTOS/databricks_managed_catalog_messages_pb2.py" "$PROTOS/databricks_uc_registry_service_pb2.py" "$PROTOS/databricks_uc_registry_messages_pb2.py" "$TEST_PROTOS/test_message_pb2.py" "$PROTOS/databricks_filesystem_service_pb2.py"

OLD_DATABRICKS="import databricks_pb2 as databricks__pb2"
NEW_DATABRICKS="from . import databricks_pb2 as databricks__pb2"
sed -i'.old' -e "s/$OLD_DATABRICKS/$NEW_DATABRICKS/g" "$PROTOS/service_pb2.py" "$PROTOS/model_registry_pb2.py" "$PROTOS/databricks_artifacts_pb2.py" "$PROTOS/mlflow_artifacts_pb2.py" "$PROTOS/internal_pb2.py" "$PROTOS/databricks_managed_catalog_service_pb2.py" "$PROTOS/databricks_managed_catalog_messages_pb2.py" "$PROTOS/databricks_uc_registry_service_pb2.py" "$PROTOS/databricks_uc_registry_messages_pb2.py" "$TEST_PROTOS/test_message_pb2.py" "$PROTOS/databricks_filesystem_service_pb2.py"

OLD_DATABRICKS_UC_REGISTRY="import databricks_uc_registry_messages_pb2 as databricks__uc__registry__messages__pb2"
NEW_DATABRICKS_UC_REGISTRY="from . import databricks_uc_registry_messages_pb2 as databricks_uc_registry_messages_pb2"
sed -i'.old' -e "s/$OLD_DATABRICKS_UC_REGISTRY/$NEW_DATABRICKS_UC_REGISTRY/g"  "$PROTOS/databricks_uc_registry_service_pb2.py" "$PROTOS/databricks_uc_registry_messages_pb2.py"  "$TEST_PROTOS/test_message_pb2.py" "$PROTOS/databricks_filesystem_service_pb2.py"

OLD_DATABRICKS_MANAGED_CATALOG="import databricks_managed_catalog_messages_pb2 as databricks__managed__catalog__messages__pb2"
NEW_DATABRICKS_MANAGED_CATALOG="from . import databricks_managed_catalog_messages_pb2 as databricks_managed_catalog_messages_pb2"
sed -i'.old' -e "s/$OLD_DATABRICKS_MANAGED_CATALOG/$NEW_DATABRICKS_MANAGED_CATALOG/g"  "$PROTOS/databricks_managed_catalog_service_pb2.py" "$PROTOS/databricks_managed_catalog_messages_pb2.py"  "$TEST_PROTOS/test_message_pb2.py"

rm "$PROTOS/databricks_pb2.py.old"
rm "$PROTOS/service_pb2.py.old"
rm "$PROTOS/model_registry_pb2.py.old"
rm "$PROTOS/databricks_artifacts_pb2.py.old"
rm "$PROTOS/mlflow_artifacts_pb2.py.old"
rm "$PROTOS/internal_pb2.py.old"
rm "$PROTOS/databricks_uc_registry_messages_pb2.py.old"
rm "$PROTOS/databricks_uc_registry_service_pb2.py.old"
rm "$PROTOS/databricks_managed_catalog_messages_pb2.py.old"
rm "$PROTOS/databricks_managed_catalog_service_pb2.py.old"
rm "$TEST_PROTOS/test_message_pb2.py.old"
rm "$PROTOS/databricks_filesystem_service_pb2.py.old"

python ./dev/proto_to_graphql/code_generator.py
