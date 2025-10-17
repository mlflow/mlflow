import json

from google.protobuf.timestamp_pb2 import Timestamp

import mlflow
from mlflow.entities import (
    AssessmentSource,
    Expectation,
    Feedback,
    Trace,
    TraceData,
    TraceInfo,
    TraceState,
)
from mlflow.entities.trace_location import (
    InferenceTableLocation,
    MlflowExperimentLocation,
    TraceLocation,
    TraceLocationType,
    UCSchemaLocation,
)
from mlflow.protos import assessments_pb2
from mlflow.protos import databricks_tracing_pb2 as pb
from mlflow.protos.assessments_pb2 import AssessmentSource as ProtoAssessmentSource
from mlflow.tracing.constant import (
    TRACE_ID_V4_PREFIX,
    TRACE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION_KEY,
    SpanAttributeKey,
)
from mlflow.tracing.utils import TraceMetadataKey, add_size_stats_to_trace_metadata
from mlflow.utils.databricks_tracing_utils import (
    assessment_to_proto,
    get_trace_id_from_assessment_proto,
    inference_table_location_to_proto,
    mlflow_experiment_location_to_proto,
    trace_from_proto,
    trace_location_from_proto,
    trace_location_to_proto,
    trace_to_proto,
    uc_schema_location_from_proto,
    uc_schema_location_to_proto,
)


def test_trace_location_to_proto_uc_schema():
    trace_location = TraceLocation.from_databricks_uc_schema(
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


def test_uc_schema_location_from_proto():
    proto = pb.UCSchemaLocation(
        catalog_name="test_catalog",
        schema_name="test_schema",
        otel_spans_table_name="test_spans",
        otel_logs_table_name="test_logs",
    )
    schema_location = uc_schema_location_from_proto(proto)
    assert schema_location.catalog_name == "test_catalog"
    assert schema_location.schema_name == "test_schema"
    assert schema_location.full_otel_spans_table_name == "test_catalog.test_schema.test_spans"
    assert schema_location.full_otel_logs_table_name == "test_catalog.test_schema.test_logs"


def test_inference_table_location_to_proto():
    table_location = InferenceTableLocation(full_table_name="test_catalog.test_schema.test_table")
    proto = inference_table_location_to_proto(table_location)
    assert proto.full_table_name == "test_catalog.test_schema.test_table"


def test_mlflow_experiment_location_to_proto():
    experiment_location = MlflowExperimentLocation(experiment_id="1234")
    proto = mlflow_experiment_location_to_proto(experiment_location)
    assert proto.experiment_id == "1234"


def test_schema_location_to_proto():
    schema_location = UCSchemaLocation(
        catalog_name="test_catalog",
        schema_name="test_schema",
    )
    schema_location._otel_spans_table_name = "test_spans"
    schema_location._otel_logs_table_name = "test_logs"
    proto = uc_schema_location_to_proto(schema_location)
    assert proto.catalog_name == "test_catalog"
    assert proto.schema_name == "test_schema"
    assert proto.otel_spans_table_name == "test_spans"
    assert proto.otel_logs_table_name == "test_logs"


def test_trace_location_from_proto_uc_schema():
    proto = pb.TraceLocation(
        type=pb.TraceLocation.TraceLocationType.UC_SCHEMA,
        uc_schema=pb.UCSchemaLocation(
            catalog_name="catalog",
            schema_name="schema",
            otel_spans_table_name="spans",
            otel_logs_table_name="logs",
        ),
    )
    trace_location = trace_location_from_proto(proto)
    assert trace_location.uc_schema.catalog_name == "catalog"
    assert trace_location.uc_schema.schema_name == "schema"
    assert trace_location.uc_schema.full_otel_spans_table_name == "catalog.schema.spans"
    assert trace_location.uc_schema.full_otel_logs_table_name == "catalog.schema.logs"


def test_trace_location_from_proto_mlflow_experiment():
    proto = pb.TraceLocation(
        type=pb.TraceLocation.TraceLocationType.MLFLOW_EXPERIMENT,
        mlflow_experiment=mlflow_experiment_location_to_proto(
            MlflowExperimentLocation(experiment_id="1234")
        ),
    )
    trace_location = trace_location_from_proto(proto)
    assert trace_location.type == TraceLocationType.MLFLOW_EXPERIMENT
    assert trace_location.mlflow_experiment.experiment_id == "1234"


def test_trace_location_from_proto_inference_table():
    proto = pb.TraceLocation(
        type=pb.TraceLocation.TraceLocationType.INFERENCE_TABLE,
        inference_table=inference_table_location_to_proto(
            InferenceTableLocation(full_table_name="test_catalog.test_schema.test_table")
        ),
    )
    trace_location = trace_location_from_proto(proto)
    assert trace_location.type == TraceLocationType.INFERENCE_TABLE
    assert trace_location.inference_table.full_table_name == "test_catalog.test_schema.test_table"


def test_trace_info_to_v4_proto():
    otel_trace_id = "2efb31387ff19263f92b2c0a61b0a8bc"
    trace_id = f"trace:/catalog.schema/{otel_trace_id}"
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_databricks_uc_schema(
            catalog_name="catalog", schema_name="schema"
        ),
        request_time=0,
        state=TraceState.OK,
        request_preview="request",
        response_preview="response",
        client_request_id="client_request_id",
        tags={"key": "value"},
    )
    proto_trace_info = trace_info.to_proto()
    assert proto_trace_info.trace_id == otel_trace_id
    assert proto_trace_info.trace_location.uc_schema.catalog_name == "catalog"
    assert proto_trace_info.trace_location.uc_schema.schema_name == "schema"
    assert proto_trace_info.state == 1
    assert proto_trace_info.request_preview == "request"
    assert proto_trace_info.response_preview == "response"
    assert proto_trace_info.client_request_id == "client_request_id"
    assert proto_trace_info.tags == {"key": "value"}
    assert len(proto_trace_info.assessments) == 0

    trace_info_from_proto = TraceInfo.from_proto(proto_trace_info)
    assert trace_info_from_proto == trace_info


def test_trace_to_proto_and_from_proto():
    with mlflow.start_span() as span:
        otel_trace_id = span.trace_id.removeprefix("tr-")
        uc_schema = "catalog.schema"
        trace_id = f"trace:/{uc_schema}/{otel_trace_id}"
        span.set_attribute(SpanAttributeKey.REQUEST_ID, trace_id)
        mlflow_span = span.to_immutable_span()

    assert mlflow_span.trace_id == trace_id
    trace = Trace(
        info=TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation.from_databricks_uc_schema(
                catalog_name="catalog", schema_name="schema"
            ),
            request_time=0,
            state=TraceState.OK,
            request_preview="request",
            response_preview="response",
            client_request_id="client_request_id",
            tags={"key": "value"},
        ),
        data=TraceData(spans=[mlflow_span]),
    )

    proto_trace_v4 = trace_to_proto(trace)

    assert proto_trace_v4.trace_info.trace_id == otel_trace_id
    assert proto_trace_v4.trace_info.trace_location.uc_schema.catalog_name == "catalog"
    assert proto_trace_v4.trace_info.trace_location.uc_schema.schema_name == "schema"
    assert len(proto_trace_v4.spans) == len(trace.data.spans)

    reconstructed_trace = trace_from_proto(proto_trace_v4, location_id="catalog.schema")

    assert reconstructed_trace.info.trace_id == trace_id
    assert reconstructed_trace.info.trace_location.uc_schema.catalog_name == "catalog"
    assert reconstructed_trace.info.trace_location.uc_schema.schema_name == "schema"
    assert len(reconstructed_trace.data.spans) == len(trace.data.spans)

    original_span = trace.data.spans[0]
    reconstructed_span = reconstructed_trace.data.spans[0]

    assert reconstructed_span.name == original_span.name
    assert reconstructed_span.span_id == original_span.span_id
    assert reconstructed_span.trace_id == original_span.trace_id
    assert reconstructed_span.inputs == original_span.inputs
    assert reconstructed_span.outputs == original_span.outputs
    assert reconstructed_span.get_attribute("custom") == original_span.get_attribute("custom")


def test_trace_from_proto_with_location_preserves_v4_trace_id():
    with mlflow.start_span() as span:
        otel_trace_id = span.trace_id.removeprefix("tr-")
        uc_schema = "catalog.schema"
        trace_id_v4 = f"{TRACE_ID_V4_PREFIX}{uc_schema}/{otel_trace_id}"
        span.set_attribute(SpanAttributeKey.REQUEST_ID, trace_id_v4)
        mlflow_span = span.to_immutable_span()

    # Create trace with v4 trace ID
    trace = Trace(
        info=TraceInfo(
            trace_id=trace_id_v4,
            trace_location=TraceLocation.from_databricks_uc_schema(
                catalog_name="catalog", schema_name="schema"
            ),
            request_time=0,
            state=TraceState.OK,
        ),
        data=TraceData(spans=[mlflow_span]),
    )

    # Convert to proto
    proto_trace = trace_to_proto(trace)

    # Reconstruct with location parameter
    reconstructed_trace = trace_from_proto(proto_trace, location_id=uc_schema)

    # Verify that all spans have the correct v4 trace_id format
    for reconstructed_span in reconstructed_trace.data.spans:
        assert reconstructed_span.trace_id == trace_id_v4
        assert reconstructed_span.trace_id.startswith(TRACE_ID_V4_PREFIX)
        # Verify the REQUEST_ID attribute is also in v4 format
        request_id = reconstructed_span.get_attribute("mlflow.traceRequestId")
        assert request_id == trace_id_v4


def test_trace_info_from_proto_handles_uc_schema_location():
    request_time = Timestamp()
    request_time.FromMilliseconds(1234567890)
    proto = pb.TraceInfo(
        trace_id="test_trace_id",
        trace_location=trace_location_to_proto(
            TraceLocation.from_databricks_uc_schema(catalog_name="catalog", schema_name="schema")
        ),
        request_preview="test request",
        response_preview="test response",
        request_time=request_time,
        state=TraceState.OK.to_proto(),
        trace_metadata={
            TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION),
            "other_key": "other_value",
        },
        tags={"test_tag": "test_value"},
    )
    trace_info = TraceInfo.from_proto(proto)
    assert trace_info.trace_location.uc_schema.catalog_name == "catalog"
    assert trace_info.trace_location.uc_schema.schema_name == "schema"
    assert trace_info.trace_metadata[TRACE_SCHEMA_VERSION_KEY] == str(TRACE_SCHEMA_VERSION)
    assert trace_info.trace_metadata["other_key"] == "other_value"
    assert trace_info.tags == {"test_tag": "test_value"}


def test_add_size_stats_to_trace_metadata_for_v4_trace():
    with mlflow.start_span() as span:
        otel_trace_id = span.trace_id.removeprefix("tr-")
        uc_schema = "catalog.schema"
        trace_id = f"trace:/{uc_schema}/{otel_trace_id}"
        span.set_attribute(SpanAttributeKey.REQUEST_ID, trace_id)
        mlflow_span = span.to_immutable_span()

    trace = Trace(
        info=TraceInfo(
            trace_id="test_trace_id",
            trace_location=TraceLocation.from_databricks_uc_schema(
                catalog_name="catalog", schema_name="schema"
            ),
            request_time=0,
            state=TraceState.OK,
            request_preview="request",
            response_preview="response",
            client_request_id="client_request_id",
            tags={"key": "value"},
        ),
        data=TraceData(spans=[mlflow_span]),
    )
    add_size_stats_to_trace_metadata(trace)
    assert TraceMetadataKey.SIZE_STATS in trace.info.trace_metadata


def test_assessment_to_proto():
    # Test with Feedback assessment
    feedback = Feedback(
        name="correctness",
        value=0.95,
        source=AssessmentSource(source_type="LLM_JUDGE", source_id="gpt-4"),
        trace_id="trace:/catalog.schema/trace123",
        metadata={"model": "gpt-4", "temperature": "0.7"},
        span_id="span456",
        rationale="The response is accurate and complete",
        overrides="old_assessment_id",
        valid=False,
    )
    feedback.assessment_id = "assessment789"

    proto_v4 = assessment_to_proto(feedback)

    # Validate proto structure
    assert isinstance(proto_v4, pb.Assessment)
    assert proto_v4.assessment_name == "correctness"
    assert proto_v4.assessment_id == "assessment789"
    assert proto_v4.span_id == "span456"
    assert proto_v4.rationale == "The response is accurate and complete"
    assert proto_v4.overrides == "old_assessment_id"
    assert proto_v4.valid is False

    # Check TraceIdentifier
    assert proto_v4.trace_id == "trace123"
    assert proto_v4.trace_location.uc_schema.catalog_name == "catalog"
    assert proto_v4.trace_location.uc_schema.schema_name == "schema"

    # Check source
    assert proto_v4.source.source_type == ProtoAssessmentSource.SourceType.Value("LLM_JUDGE")
    assert proto_v4.source.source_id == "gpt-4"

    # Check metadata
    assert proto_v4.metadata["model"] == "gpt-4"
    assert proto_v4.metadata["temperature"] == "0.7"

    # Check feedback value
    assert proto_v4.HasField("feedback")
    assert proto_v4.feedback.value.number_value == 0.95

    # Test with Expectation assessment
    expectation = Expectation(
        name="expected_answer",
        value={"answer": "Paris", "confidence": 0.99},
        source=AssessmentSource(source_type="HUMAN", source_id="user@example.com"),
        trace_id="trace:/main.default/trace789",
        metadata={"question": "What is the capital of France?"},
        span_id="span111",
    )
    expectation.assessment_id = "exp_assessment123"

    proto_v4_exp = assessment_to_proto(expectation)

    assert isinstance(proto_v4_exp, pb.Assessment)
    assert proto_v4_exp.assessment_name == "expected_answer"
    assert proto_v4_exp.assessment_id == "exp_assessment123"
    assert proto_v4_exp.span_id == "span111"

    # Check TraceIdentifier for expectation
    assert proto_v4_exp.trace_id == "trace789"
    assert proto_v4_exp.trace_location.uc_schema.catalog_name == "main"
    assert proto_v4_exp.trace_location.uc_schema.schema_name == "default"

    # Check expectation value
    assert proto_v4_exp.HasField("expectation")
    assert proto_v4_exp.expectation.HasField("serialized_value")
    assert json.loads(proto_v4_exp.expectation.serialized_value.value) == {
        "answer": "Paris",
        "confidence": 0.99,
    }


def test_get_trace_id_from_assessment_proto():
    proto = pb.Assessment(
        trace_id="1234",
        trace_location=trace_location_to_proto(
            TraceLocation.from_databricks_uc_schema(catalog_name="catalog", schema_name="schema")
        ),
    )
    assert get_trace_id_from_assessment_proto(proto) == "trace:/catalog.schema/1234"

    proto = assessments_pb2.Assessment(
        trace_id="tr-123",
    )
    assert get_trace_id_from_assessment_proto(proto) == "tr-123"
