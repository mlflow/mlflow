"""Tests for Span <-> OTel protobuf conversion."""

import pytest
from opentelemetry.proto.trace.v1.trace_pb2 import Span as OTelProtoSpan
from opentelemetry.proto.trace.v1.trace_pb2 import Status as OTelProtoStatus
from opentelemetry.sdk.trace import Event as OTelEvent
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.trace import Status as OTelStatus
from opentelemetry.trace import StatusCode as OTelStatusCode

from mlflow.entities.span import Span
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatus, SpanStatusCode
from mlflow.tracing.utils import build_otel_context


@pytest.fixture
def sample_otel_span():
    """Create a sample OTelReadableSpan for testing."""
    return OTelReadableSpan(
        name="test_span",
        context=build_otel_context(
            trace_id=0x12345678901234567890123456789012,
            span_id=0x1234567890123456,
        ),
        parent=build_otel_context(
            trace_id=0x12345678901234567890123456789012,
            span_id=0x0987654321098765,
        ),
        start_time=1000000000,
        end_time=2000000000,
        attributes={
            "mlflow.traceRequestId": '{"test": "trace_id"}',
            "mlflow.spanType": "LLM",
            "mlflow.spanInputs": '{"prompt": "Hello"}',
            "mlflow.spanOutputs": '{"response": "Hi"}',
            "custom_attr": '{"key": "value"}',
        },
        status=OTelStatus(OTelStatusCode.OK, "Success"),
        events=[
            OTelEvent(
                name="event1",
                timestamp=1500000000,
                attributes={"event_key": "event_value"},
            )
        ],
    )


@pytest.mark.parametrize(
    ("status_code", "status_desc", "expected_code", "expected_desc"),
    [
        # OTel only keeps description for ERROR status
        (OTelStatusCode.OK, "Success", SpanStatusCode.OK, ""),
        (OTelStatusCode.ERROR, "Failed", SpanStatusCode.ERROR, "Failed"),
        (OTelStatusCode.UNSET, "", SpanStatusCode.UNSET, ""),
    ],
)
def test_status_conversion(status_code, status_desc, expected_code, expected_desc):
    """Test status conversion between OTel and MLflow."""
    # Create OTel status
    otel_status = OTelStatus(status_code, status_desc)

    # Convert to MLflow status
    mlflow_status = SpanStatus.from_otel_status(otel_status)
    assert mlflow_status.status_code == expected_code
    # OTel SDK clears description for non-ERROR statuses
    assert mlflow_status.description == (status_desc if status_code == OTelStatusCode.ERROR else "")

    # Convert back to OTel status
    converted_status = mlflow_status.to_otel_status()
    assert converted_status.status_code == status_code
    # Description is only preserved for ERROR status
    assert converted_status.description == expected_desc


@pytest.mark.parametrize(
    ("proto_code", "expected_mlflow_code", "expected_otel_code"),
    [
        (OTelProtoStatus.STATUS_CODE_OK, SpanStatusCode.OK, OTelStatusCode.OK),
        (OTelProtoStatus.STATUS_CODE_ERROR, SpanStatusCode.ERROR, OTelStatusCode.ERROR),
        (OTelProtoStatus.STATUS_CODE_UNSET, SpanStatusCode.UNSET, OTelStatusCode.UNSET),
    ],
)
def test_proto_status_conversion(proto_code, expected_mlflow_code, expected_otel_code):
    """Test status conversion from OTel protobuf."""
    # Create proto status
    proto_status = OTelProtoStatus()
    proto_status.code = proto_code
    proto_status.message = "test message"

    # Convert to MLflow status
    mlflow_status = SpanStatus.from_otel_proto_status(proto_status)
    assert mlflow_status.status_code == expected_mlflow_code
    assert mlflow_status.description == "test message"

    # Convert to OTel proto status
    converted_proto = mlflow_status.to_otel_proto_status()
    assert converted_proto.code == proto_code
    assert converted_proto.message == "test message"


@pytest.mark.parametrize(
    "attributes",
    [
        # Empty attributes
        {},
        # String attributes
        {"str_key": "str_value"},
        # Numeric attributes
        {"int_key": 42, "float_key": 3.14},
        # Boolean attributes
        {"bool_key": True, "false_key": False},
        # Bytes attributes
        {"bytes_key": b"binary_data"},
        # List attributes
        {"list_str": ["a", "b", "c"], "list_int": [1, 2, 3], "list_float": [1.1, 2.2]},
        # Dict attributes
        {"dict_key": {"nested": "value", "number": 123}},
        # Mixed complex attributes
        {
            "complex": {
                "nested_list": [1, "two", 3.0],
                "nested_dict": {"deep": {"deeper": "value"}},
            },
            "simple": "string",
        },
    ],
)
def test_attribute_conversion(attributes):
    """Test attribute conversion with various data types."""
    from opentelemetry.proto.common.v1.common_pb2 import KeyValue

    from mlflow.tracing.utils.otlp import decode_otel_proto_anyvalue, set_otel_proto_anyvalue

    # Convert attributes to proto format
    proto_attrs = []
    for key, value in attributes.items():
        kv = KeyValue()
        kv.key = key
        set_otel_proto_anyvalue(kv.value, value)
        proto_attrs.append(kv)

    # Decode back and verify
    decoded = {}
    for kv in proto_attrs:
        decoded[kv.key] = decode_otel_proto_anyvalue(kv.value)

    assert decoded == attributes


def test_span_to_otel_proto_conversion(sample_otel_span):
    """Test converting MLflow Span to OTel protobuf."""
    # Create MLflow span from OTel span
    mlflow_span = Span(sample_otel_span)

    # Convert to OTel proto
    otel_proto = mlflow_span._to_otel_proto()

    # Verify basic fields
    assert otel_proto.name == "test_span"
    assert otel_proto.start_time_unix_nano == 1000000000
    assert otel_proto.end_time_unix_nano == 2000000000

    # Verify IDs (should be in bytes format)
    assert len(otel_proto.trace_id) == 16  # 128-bit trace ID
    assert len(otel_proto.span_id) == 8  # 64-bit span ID
    assert len(otel_proto.parent_span_id) == 8

    # Verify status
    assert otel_proto.status.code == OTelProtoStatus.STATUS_CODE_OK
    # OTel SDK clears description for non-ERROR statuses
    assert otel_proto.status.message == ""

    # Verify attributes exist
    assert len(otel_proto.attributes) == 5
    attr_keys = {attr.key for attr in otel_proto.attributes}
    assert "mlflow.spanType" in attr_keys
    assert "custom_attr" in attr_keys

    # Verify events
    assert len(otel_proto.events) == 1
    assert otel_proto.events[0].name == "event1"
    assert otel_proto.events[0].time_unix_nano == 1500000000


def test_span_from_otel_proto_conversion():
    """Test converting OTel protobuf to MLflow Span."""
    # Create OTel proto span
    otel_proto = OTelProtoSpan()
    otel_proto.trace_id = bytes.fromhex("12345678901234567890123456789012")
    otel_proto.span_id = bytes.fromhex("1234567890123456")
    otel_proto.parent_span_id = bytes.fromhex("0987654321098765")
    otel_proto.name = "proto_span"
    otel_proto.start_time_unix_nano = 1000000000
    otel_proto.end_time_unix_nano = 2000000000

    # Add status
    otel_proto.status.code = OTelProtoStatus.STATUS_CODE_ERROR
    otel_proto.status.message = "Error occurred"

    # Add attributes
    from mlflow.tracing.utils.otlp import set_otel_proto_anyvalue

    attr1 = otel_proto.attributes.add()
    attr1.key = "mlflow.traceRequestId"
    set_otel_proto_anyvalue(attr1.value, '{"request": "id"}')

    attr2 = otel_proto.attributes.add()
    attr2.key = "mlflow.spanType"
    set_otel_proto_anyvalue(attr2.value, "CHAIN")

    attr3 = otel_proto.attributes.add()
    attr3.key = "custom"
    set_otel_proto_anyvalue(attr3.value, {"nested": {"value": 123}})

    # Add event
    event = otel_proto.events.add()
    event.name = "test_event"
    event.time_unix_nano = 1500000000
    event_attr = event.attributes.add()
    event_attr.key = "event_data"
    set_otel_proto_anyvalue(event_attr.value, "event_value")

    # Convert to MLflow span
    mlflow_span = Span._from_otel_proto(otel_proto)

    # Verify basic fields
    assert mlflow_span.name == "proto_span"
    assert mlflow_span.start_time_ns == 1000000000
    assert mlflow_span.end_time_ns == 2000000000

    # Verify IDs
    assert mlflow_span.span_id == "1234567890123456"
    assert mlflow_span.parent_id == "0987654321098765"

    # Verify status
    assert mlflow_span.status.status_code == SpanStatusCode.ERROR
    assert mlflow_span.status.description == "Error occurred"

    # Verify attributes
    assert mlflow_span.span_type == "CHAIN"
    assert mlflow_span.get_attribute("custom") == {"nested": {"value": 123}}

    # Verify events
    assert len(mlflow_span.events) == 1
    assert mlflow_span.events[0].name == "test_event"
    assert mlflow_span.events[0].timestamp == 1500000000
    assert mlflow_span.events[0].attributes["event_data"] == "event_value"


def test_roundtrip_conversion(sample_otel_span):
    """Test that conversion roundtrip preserves data."""
    # Start with OTel span -> MLflow span
    mlflow_span = Span(sample_otel_span)

    # Convert to OTel proto
    otel_proto = mlflow_span._to_otel_proto()

    # Convert back to MLflow span
    roundtrip_span = Span._from_otel_proto(otel_proto)

    # Verify key fields are preserved
    assert roundtrip_span.name == mlflow_span.name
    assert roundtrip_span.span_id == mlflow_span.span_id
    assert roundtrip_span.parent_id == mlflow_span.parent_id
    assert roundtrip_span.start_time_ns == mlflow_span.start_time_ns
    assert roundtrip_span.end_time_ns == mlflow_span.end_time_ns
    assert roundtrip_span.status.status_code == mlflow_span.status.status_code
    assert roundtrip_span.status.description == mlflow_span.status.description

    # Verify attributes
    assert roundtrip_span.span_type == mlflow_span.span_type
    assert roundtrip_span.inputs == mlflow_span.inputs
    assert roundtrip_span.outputs == mlflow_span.outputs

    # Verify events
    assert len(roundtrip_span.events) == len(mlflow_span.events)
    for orig_event, rt_event in zip(mlflow_span.events, roundtrip_span.events):
        assert rt_event.name == orig_event.name
        assert rt_event.timestamp == orig_event.timestamp
        assert rt_event.attributes == orig_event.attributes


@pytest.mark.parametrize(
    "event_attrs",
    [
        {},
        {"simple": "string"},
        {"number": 42, "float": 3.14, "bool": True},
        {"list": [1, 2, 3], "dict": {"nested": "value"}},
    ],
)
def test_span_event_otel_conversion(event_attrs):
    """Test SpanEvent to OTel proto conversion."""
    # Create span event
    event = SpanEvent(
        name="test_event",
        timestamp=1234567890,
        attributes=event_attrs,
    )

    # Convert to OTel proto
    otel_proto_event = event._to_otel_proto()

    # Verify fields
    assert otel_proto_event.name == "test_event"
    assert otel_proto_event.time_unix_nano == 1234567890

    # Verify attributes
    from mlflow.tracing.utils.otlp import decode_otel_proto_anyvalue

    decoded_attrs = {}
    for attr in otel_proto_event.attributes:
        decoded_attrs[attr.key] = decode_otel_proto_anyvalue(attr.value)

    assert decoded_attrs == event_attrs
