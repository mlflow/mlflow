"""
Tests for TraceArchiveConfiguration entity.
"""

import pytest

from mlflow.entities.databricks_trace_storage_config import (
    DatabricksTraceDeltaStorageConfig,
)
from mlflow.exceptions import MlflowException


def test_trace_storage_configuration_entity():
    """Test TraceArchiveConfiguration entity creation and conversion."""
    destination = DatabricksTraceDeltaStorageConfig(
        experiment_id="12345",
        spans_table_name="catalog.schema.spans",
        logs_table_name="catalog.schema.events",
        spans_schema_version="v1",
        logs_schema_version="v1",
    )

    # Test to_dict conversion
    dict_result = destination.to_dict()
    assert dict_result["experiment_id"] == "12345"
    assert dict_result["spans_table_name"] == "catalog.schema.spans"
    assert dict_result["logs_table_name"] == "catalog.schema.events"
    assert dict_result["spans_schema_version"] == "v1"
    assert dict_result["logs_schema_version"] == "v1"

    # Test from_dict conversion
    reconstructed = DatabricksTraceDeltaStorageConfig.from_dict(dict_result)
    assert reconstructed.experiment_id == destination.experiment_id
    assert reconstructed.spans_table_name == destination.spans_table_name
    assert reconstructed.logs_table_name == destination.logs_table_name
    assert reconstructed.spans_schema_version == destination.spans_schema_version
    assert reconstructed.logs_schema_version == destination.logs_schema_version


def test_trace_storage_configuration_from_proto_validation():
    """Test that from_proto validates only experiment locations are supported."""
    from mlflow.protos.databricks_trace_server_pb2 import TraceDestination as ProtoTraceDestination
    from mlflow.protos.databricks_trace_server_pb2 import TraceLocation as ProtoTraceLocation

    # Test with valid experiment location
    proto_trace_location = ProtoTraceLocation()
    proto_trace_location.type = ProtoTraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
    proto_trace_location.mlflow_experiment.experiment_id = "12345"

    proto_destination = ProtoTraceDestination()
    proto_destination.trace_location.CopyFrom(proto_trace_location)
    proto_destination.spans_table_name = "catalog.schema.spans"
    proto_destination.logs_table_name = "catalog.schema.events"
    proto_destination.spans_schema_version = "v1"
    proto_destination.logs_schema_version = "v1"

    config = DatabricksTraceDeltaStorageConfig.from_proto(proto_destination)
    assert config.experiment_id == "12345"

    # Test with invalid inference table location
    proto_trace_location_invalid = ProtoTraceLocation()
    proto_trace_location_invalid.type = ProtoTraceLocation.TraceLocationType.INFERENCE_TABLE
    proto_trace_location_invalid.inference_table.full_table_name = "catalog.schema.table"

    proto_destination_invalid = ProtoTraceDestination()
    proto_destination_invalid.trace_location.CopyFrom(proto_trace_location_invalid)
    proto_destination_invalid.spans_table_name = "catalog.schema.spans"
    proto_destination_invalid.logs_table_name = "catalog.schema.events"
    proto_destination_invalid.spans_schema_version = "v1"
    proto_destination_invalid.logs_schema_version = "v1"

    with pytest.raises(
        MlflowException, match="TraceArchiveConfiguration only supports MLflow experiments"
    ):
        DatabricksTraceDeltaStorageConfig.from_proto(proto_destination_invalid)
