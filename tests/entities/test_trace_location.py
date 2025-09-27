import pytest

from mlflow.entities.trace_location import (
    InferenceTableLocation,
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
    UCSchemaLocation,
)
from mlflow.exceptions import MlflowException
from mlflow.protos import service_pb2 as pb


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


def test_trace_location_from_proto_mlflow_experiment():
    proto = pb.TraceLocation(
        type=pb.TraceLocation.TraceLocationType.MLFLOW_EXPERIMENT,
        mlflow_experiment=pb.TraceLocation.MlflowExperimentLocation(experiment_id="1234"),
    )
    trace_location = TraceLocation.from_proto(proto)
    assert trace_location.type == TraceLocationType.MLFLOW_EXPERIMENT
    assert trace_location.mlflow_experiment.experiment_id == "1234"


def test_trace_location_from_proto_inference_table():
    proto = pb.TraceLocation(
        type=pb.TraceLocation.TraceLocationType.INFERENCE_TABLE,
        inference_table=pb.TraceLocation.InferenceTableLocation(
            full_table_name="test_catalog.test_schema.test_table"
        ),
    )
    trace_location = TraceLocation.from_proto(proto)
    assert trace_location.type == TraceLocationType.INFERENCE_TABLE
    assert trace_location.inference_table.full_table_name == "test_catalog.test_schema.test_table"
