from mlflow.entities.trace_location import (
    InferenceTableLocation,
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
    UCSchemaLocation,
)
from mlflow.protos import databricks_tracing_pb2 as pb
from mlflow.utils.databricks_tracing_utils import (
    inference_table_location_to_proto,
    mlflow_experiment_location_to_proto,
    trace_location_to_proto,
    uc_schema_location_to_proto,
)


def test_trace_location_to_proto_uc_schema():
    trace_location = TraceLocation.from_uc_schema(
        catalog_name="test_catalog", schema_name="test_schema"
    )
    proto = trace_location_to_proto(trace_location)
    assert proto.type == pb.TraceLocation.TraceLocationType.UC_SCHEMA
    assert proto.uc_schema.catalog_name == "test_catalog"
    assert proto.uc_schema.schema_name == "test_schema"


def test_trace_location_to_proto_mlflow_experiment():
    trace_location = TraceLocation.from_experiment_id(experiment_id="1234")
    proto = trace_location_to_proto(trace_location)
    assert proto.type == pb.TraceLocation.TraceLocationType.MLFLOW_EXPERIMENT
    assert proto.mlflow_experiment.experiment_id == "1234"


def test_trace_location_to_proto_inference_table():
    trace_location = TraceLocation(
        type=TraceLocationType.INFERENCE_TABLE,
        inference_table=InferenceTableLocation(
            full_table_name="test_catalog.test_schema.test_table"
        ),
    )
    proto = trace_location_to_proto(trace_location)
    assert proto.type == pb.TraceLocation.TraceLocationType.INFERENCE_TABLE
    assert proto.inference_table.full_table_name == "test_catalog.test_schema.test_table"


def test_uc_schema_location_to_proto():
    schema_location = UCSchemaLocation(catalog_name="test_catalog", schema_name="test_schema")
    proto = uc_schema_location_to_proto(schema_location)
    assert proto.catalog_name == "test_catalog"
    assert proto.schema_name == "test_schema"


def test_inference_table_location_to_proto():
    table_location = InferenceTableLocation(full_table_name="test_catalog.test_schema.test_table")
    proto = inference_table_location_to_proto(table_location)
    assert proto.full_table_name == "test_catalog.test_schema.test_table"


def test_mlflow_experiment_location_to_proto():
    experiment_location = MlflowExperimentLocation(experiment_id="1234")
    proto = mlflow_experiment_location_to_proto(experiment_location)
    assert proto.experiment_id == "1234"
