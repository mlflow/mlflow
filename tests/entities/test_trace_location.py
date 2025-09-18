import pytest

from mlflow.entities.trace_location import (
    InferenceTableLocation,
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
    UCSchemaLocation,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import UCSchemaLocation as UCSchemaLocationProto


def test_trace_location():
    trace_location = TraceLocation(
        type=TraceLocationType.MLFLOW_EXPERIMENT,
        mlflow_experiment=MlflowExperimentLocation(experiment_id="123"),
    )
    assert trace_location.type == TraceLocationType.MLFLOW_EXPERIMENT
    assert trace_location.mlflow_experiment.experiment_id == "123"

    trace_location = TraceLocation(
        type=TraceLocationType.INFERENCE_TABLE,
        inference_table=InferenceTableLocation(full_table_name="a.b.c"),
    )
    assert trace_location.type == TraceLocationType.INFERENCE_TABLE
    assert trace_location.inference_table.full_table_name == "a.b.c"

    trace_location = TraceLocation(
        type=TraceLocationType.UC_SCHEMA,
        uc_schema=UCSchemaLocation(catalog_name="a", schema_name="b"),
    )
    assert trace_location.type == TraceLocationType.UC_SCHEMA
    assert trace_location.uc_schema.catalog_name == "a"
    assert trace_location.uc_schema.schema_name == "b"

    from_proto = TraceLocation.from_proto(trace_location.to_proto())
    assert from_proto == trace_location

    with pytest.raises(
        MlflowException,
        match="Only one of mlflow_experiment, inference_table, or uc_schema can be provided",
    ):
        TraceLocation(
            type=TraceLocationType.TRACE_LOCATION_TYPE_UNSPECIFIED,
            mlflow_experiment=MlflowExperimentLocation(experiment_id="123"),
            inference_table=InferenceTableLocation(full_table_name="a.b.c"),
            uc_schema=UCSchemaLocation(catalog_name="a", schema_name="b"),
        )


def test_trace_location_mismatch():
    with pytest.raises(
        MlflowException, match="Trace location .+ does not match the provided location"
    ):
        TraceLocation(
            type=TraceLocationType.INFERENCE_TABLE,
            mlflow_experiment=MlflowExperimentLocation(experiment_id="123"),
        )

    with pytest.raises(
        MlflowException, match="Trace location .+ does not match the provided location"
    ):
        TraceLocation(
            type=TraceLocationType.MLFLOW_EXPERIMENT,
            inference_table=InferenceTableLocation(full_table_name="a.b.c"),
        )

    with pytest.raises(
        MlflowException, match="Trace location .+ does not match the provided location"
    ):
        TraceLocation(
            type=TraceLocationType.INFERENCE_TABLE,
            uc_schema=UCSchemaLocation(catalog_name="a", schema_name="b"),
        )


def test_ucschema_location_from_and_to_proto():
    proto = UCSchemaLocationProto(catalog_name="a", schema_name="b")
    location = UCSchemaLocation.from_proto(proto)
    assert location.catalog_name == "a"
    assert location.schema_name == "b"
    assert location.to_proto() == proto

    proto = UCSchemaLocationProto(
        catalog_name="a", schema_name="b", otel_spans_table_name="c", otel_logs_table_name="d"
    )
    location = UCSchemaLocation.from_proto(proto)
    assert location.catalog_name == "a"
    assert location.schema_name == "b"
    assert location.otel_spans_table_name == "c"
    assert location.otel_logs_table_name == "d"
    assert location.to_proto() == proto
