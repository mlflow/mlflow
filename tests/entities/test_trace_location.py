import pytest

from mlflow.entities.trace_location import (
    InferenceTableLocation,
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
    UCSchemaLocation,
    UcTablePrefixLocation,
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
        match="Only one of mlflow_experiment, inference_table, uc_schema, or uc_table_prefix can be provided",
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


def test_trace_location_from_v4_proto_mlflow_experiment():
    proto = pb.TraceLocation(
        type=pb.TraceLocation.TraceLocationType.MLFLOW_EXPERIMENT,
        mlflow_experiment=pb.TraceLocation.MlflowExperimentLocation(experiment_id="1234"),
    )
    trace_location = TraceLocation.from_proto(proto)
    assert trace_location.type == TraceLocationType.MLFLOW_EXPERIMENT
    assert trace_location.mlflow_experiment.experiment_id == "1234"


def test_trace_location_from_v4_proto_inference_table():
    proto = pb.TraceLocation(
        type=pb.TraceLocation.TraceLocationType.INFERENCE_TABLE,
        inference_table=pb.TraceLocation.InferenceTableLocation(
            full_table_name="test_catalog.test_schema.test_table"
        ),
    )
    trace_location = TraceLocation.from_proto(proto)
    assert trace_location.type == TraceLocationType.INFERENCE_TABLE
    assert trace_location.inference_table.full_table_name == "test_catalog.test_schema.test_table"


def test_uc_schema_location_full_otel_spans_table_name():
    uc_schema = UCSchemaLocation(
        catalog_name="test_catalog",
        schema_name="test_schema",
    )
    uc_schema._otel_spans_table_name = "otel_spans"
    assert uc_schema.full_otel_spans_table_name == "test_catalog.test_schema.otel_spans"


def test_uc_table_prefix_location():
    uc_table_prefix = UcTablePrefixLocation(
        catalog_name="test_catalog",
        schema_name="test_schema",
        table_prefix="trace_",
    )
    assert uc_table_prefix.catalog_name == "test_catalog"
    assert uc_table_prefix.schema_name == "test_schema"
    assert uc_table_prefix.table_prefix == "trace_"
    assert uc_table_prefix.full_table_prefix == "test_catalog.test_schema.trace_"


def test_uc_table_prefix_location_to_dict():
    uc_table_prefix = UcTablePrefixLocation(
        catalog_name="test_catalog",
        schema_name="test_schema",
        table_prefix="my_prefix_",
    )
    d = uc_table_prefix.to_dict()
    assert d == {
        "catalog_name": "test_catalog",
        "schema_name": "test_schema",
        "table_prefix": "my_prefix_",
    }


def test_uc_table_prefix_location_from_dict():
    d = {
        "catalog_name": "test_catalog",
        "schema_name": "test_schema",
        "table_prefix": "my_prefix_",
    }
    uc_table_prefix = UcTablePrefixLocation.from_dict(d)
    assert uc_table_prefix.catalog_name == "test_catalog"
    assert uc_table_prefix.schema_name == "test_schema"
    assert uc_table_prefix.table_prefix == "my_prefix_"
    assert uc_table_prefix.full_table_prefix == "test_catalog.test_schema.my_prefix_"


def test_trace_location_uc_table_prefix():
    trace_location = TraceLocation(
        type=TraceLocationType.UC_TABLE_PREFIX,
        uc_table_prefix=UcTablePrefixLocation(
            catalog_name="catalog", schema_name="schema", table_prefix="prefix_"
        ),
    )
    assert trace_location.type == TraceLocationType.UC_TABLE_PREFIX
    assert trace_location.uc_table_prefix.catalog_name == "catalog"
    assert trace_location.uc_table_prefix.schema_name == "schema"
    assert trace_location.uc_table_prefix.table_prefix == "prefix_"


def test_trace_location_from_databricks_uc_table_prefix():
    trace_location = TraceLocation.from_databricks_uc_table_prefix(
        catalog_name="catalog", schema_name="schema", table_prefix="prefix_"
    )
    assert trace_location.type == TraceLocationType.UC_TABLE_PREFIX
    assert trace_location.uc_table_prefix.catalog_name == "catalog"
    assert trace_location.uc_table_prefix.schema_name == "schema"
    assert trace_location.uc_table_prefix.table_prefix == "prefix_"
    assert trace_location.uc_table_prefix.full_table_prefix == "catalog.schema.prefix_"


def test_trace_location_uc_table_prefix_to_from_dict():
    trace_location = TraceLocation(
        type=TraceLocationType.UC_TABLE_PREFIX,
        uc_table_prefix=UcTablePrefixLocation(
            catalog_name="cat", schema_name="sch", table_prefix="pre_"
        ),
    )
    d = trace_location.to_dict()
    assert d == {
        "type": "UC_TABLE_PREFIX",
        "uc_table_prefix": {
            "catalog_name": "cat",
            "schema_name": "sch",
            "table_prefix": "pre_",
        },
    }

    from_dict = TraceLocation.from_dict(d)
    assert from_dict.type == TraceLocationType.UC_TABLE_PREFIX
    assert from_dict.uc_table_prefix.catalog_name == "cat"
    assert from_dict.uc_table_prefix.schema_name == "sch"
    assert from_dict.uc_table_prefix.table_prefix == "pre_"


def test_trace_location_uc_table_prefix_type_mismatch():
    with pytest.raises(
        MlflowException, match="Trace location .+ does not match the provided location"
    ):
        TraceLocation(
            type=TraceLocationType.UC_SCHEMA,
            uc_table_prefix=UcTablePrefixLocation(
                catalog_name="a", schema_name="b", table_prefix="c"
            ),
        )
