import pytest

from mlflow.entities.trace_location import (
    InferenceTableLocation,
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
)
from mlflow.exceptions import MlflowException


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
        MlflowException, match="Only one of mlflow_experiment or inference_table can be provided"
    ):
        TraceLocation(
            type=TraceLocationType.TRACE_LOCATION_TYPE_UNSPECIFIED,
            mlflow_experiment=MlflowExperimentLocation(experiment_id="123"),
            inference_table=InferenceTableLocation(full_table_name="a.b.c"),
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
