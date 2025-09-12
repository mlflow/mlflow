import json
from datetime import datetime

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.proto.trace.v1.trace_pb2 import Span as OTelProtoSpan
from opentelemetry.proto.trace.v1.trace_pb2 import Status as OTelProtoStatus
from opentelemetry.sdk.trace import Event as OTelEvent
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.trace import Status as OTelStatus
from opentelemetry.trace import StatusCode as OTelStatusCode

import mlflow
from mlflow.entities import LiveSpan, Span, SpanEvent, SpanStatus, SpanStatusCode, SpanType
from mlflow.entities.span import NoOpSpan, create_mlflow_span
from mlflow.exceptions import MlflowException
from mlflow.tracing.provider import _get_tracer, trace_disabled
from mlflow.tracing.utils import build_otel_context, encode_span_id, encode_trace_id


def test_create_live_span():
    trace_id = "tr-12345"

    tracer = _get_tracer("test")
    with tracer.start_as_current_span("parent") as parent_span:
        span = create_mlflow_span(parent_span, trace_id=trace_id, span_type=SpanType.LLM)
        assert isinstance(span, LiveSpan)
        assert span.trace_id == trace_id
        assert span._trace_id == encode_trace_id(parent_span.context.trace_id)
        assert span.span_id == encode_span_id(parent_span.context.span_id)
        assert span.name == "parent"
        assert span.start_time_ns == parent_span.start_time
        assert span.end_time_ns is None
        assert span.parent_id is None

        span.set_inputs({"input": 1})
        span.set_outputs(2)
        assert span.inputs == {"input": 1}
        assert span.outputs == 2

        span.set_attribute("key", 3)
        assert span.get_attribute("key") == 3

        # non-serializable value should be stored as string
        non_serializable = datetime.now()
        span.set_attribute("non_serializable", non_serializable)
        assert span.get_attribute("non_serializable") == str(non_serializable)
        assert parent_span._attributes == {
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanInputs": '{"input": 1}',
            "mlflow.spanOutputs": "2",
            "mlflow.spanType": '"LLM"',
            "key": "3",
            "non_serializable": json.dumps(str(non_serializable)),
        }

        span.set_status("OK")
        assert span.status == SpanStatus(SpanStatusCode.OK)

        span.add_event(SpanEvent("test_event", timestamp=99999, attributes={"foo": "bar"}))
        assert len(span.events) == 1
        assert span.events[0].name == "test_event"
        assert span.events[0].timestamp == 99999
        assert span.events[0].attributes == {"foo": "bar"}

        # Test child span
        with tracer.start_as_current_span("child") as child_span:
            span = create_mlflow_span(child_span, trace_id=trace_id)
            assert isinstance(span, LiveSpan)
            assert span.name == "child"
            assert span.parent_id == encode_span_id(parent_span.context.span_id)


def test_create_non_live_span():
    trace_id = "tr-12345"
    parent_span_context = trace_api.SpanContext(
        trace_id=12345, span_id=111, is_remote=False, trace_flags=trace_api.TraceFlags(1)
    )
    readable_span = OTelReadableSpan(
        name="test",
        context=trace_api.SpanContext(
            trace_id=12345, span_id=222, is_remote=False, trace_flags=trace_api.TraceFlags(1)
        ),
        parent=parent_span_context,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanInputs": '{"input": 1, "nested": {"foo": "bar"}}',
            "mlflow.spanOutputs": "2",
            "key": "3",
        },
        start_time=99999,
        end_time=100000,
    )
    span = create_mlflow_span(readable_span, trace_id)

    assert isinstance(span, Span)
    assert not isinstance(span, LiveSpan)
    assert not isinstance(span, NoOpSpan)
    assert span.trace_id == trace_id
    assert span._trace_id == encode_trace_id(12345)
    assert span.span_id == encode_span_id(222)
    assert span.name == "test"
    assert span.start_time_ns == 99999
    assert span.end_time_ns == 100000
    assert span.parent_id == encode_span_id(111)
    assert span.inputs == {"input": 1, "nested": {"foo": "bar"}}
    assert span.outputs == 2
    assert span.status == SpanStatus(SpanStatusCode.UNSET, description="")
    assert span.get_attribute("key") == 3

    # Non-live span should not implement setter methods
    with pytest.raises(AttributeError, match="set_inputs"):
        span.set_inputs({"input": 1})


def test_create_noop_span():
    trace_id = "tr-12345"

    @trace_disabled
    def f():
        tracer = _get_tracer("test")
        with tracer.start_as_current_span("span") as otel_span:
            span = create_mlflow_span(otel_span, trace_id=trace_id)
        assert isinstance(span, NoOpSpan)

    # create from None
    span = create_mlflow_span(None, trace_id=trace_id)
    assert isinstance(span, NoOpSpan)


def test_create_raise_for_invalid_otel_span():
    with pytest.raises(MlflowException, match=r"The `otel_span` argument must be"):
        create_mlflow_span(otel_span=123, trace_id="tr-12345")


@pytest.mark.parametrize(
    "status",
    [SpanStatus("OK"), SpanStatus(SpanStatusCode.ERROR, "Error!"), "OK", "ERROR"],
)
def test_set_status(status):
    with mlflow.start_span("test_span") as span:
        span.set_status(status)

    assert isinstance(span.status, SpanStatus)


def test_set_status_raise_for_invalid_value():
    with mlflow.start_span("test_span") as span:
        with pytest.raises(MlflowException, match=r"INVALID is not a valid SpanStatusCode value."):
            span.set_status("INVALID")


def test_dict_conversion():
    with mlflow.start_span("parent"):
        with mlflow.start_span("child", span_type=SpanType.LLM) as span:
            span.set_inputs({"input": 1})
            span.set_outputs(2)
            span.set_attribute("key", 3)
            span.set_status("OK")
            span.add_event(SpanEvent("test_event", timestamp=0, attributes={"foo": "bar"}))

    span_dict = span.to_dict()
    recovered_span = Span.from_dict(span_dict)

    assert span.trace_id == recovered_span.trace_id
    assert span._trace_id == recovered_span._trace_id
    assert span.span_id == recovered_span.span_id
    assert span.name == recovered_span.name
    assert span.span_type == recovered_span.span_type
    assert span.start_time_ns == recovered_span.start_time_ns
    assert span.end_time_ns == recovered_span.end_time_ns
    assert span.parent_id == recovered_span.parent_id
    assert span.status == recovered_span.status
    assert span.inputs == recovered_span.inputs
    assert span.outputs == recovered_span.outputs
    assert span.attributes == recovered_span.attributes
    assert span.events == recovered_span.events

    # Loaded span should not implement setter methods
    with pytest.raises(AttributeError, match="set_status"):
        recovered_span.set_status("OK")


def test_dict_conversion_with_exception_event():
    with pytest.raises(ValueError, match="Test exception"):
        with mlflow.start_span("test") as span:
            raise ValueError("Test exception")

    span_dict = span.to_dict()
    recovered_span = Span.from_dict(span_dict)

    assert span.request_id == recovered_span.request_id
    assert span._trace_id == recovered_span._trace_id
    assert span.span_id == recovered_span.span_id
    assert span.name == recovered_span.name
    assert span.span_type == recovered_span.span_type
    assert span.start_time_ns == recovered_span.start_time_ns
    assert span.end_time_ns == recovered_span.end_time_ns
    assert span.parent_id == recovered_span.parent_id
    assert span.status == recovered_span.status
    assert span.inputs == recovered_span.inputs
    assert span.outputs == recovered_span.outputs
    assert span.attributes == recovered_span.attributes
    assert span.events == recovered_span.events

    # Loaded span should not implement setter methods
    with pytest.raises(AttributeError, match="set_status"):
        recovered_span.set_status("OK")


def test_from_v2_dict():
    span = Span.from_dict(
        {
            "name": "test",
            "context": {
                "span_id": "8a90fc46e65ea5a4",
                "trace_id": "0125978dc5c5a9456d7ca9ef1f7cf4af",
            },
            "parent_id": None,
            "start_time": 1738662897576578992,
            "end_time": 1738662899068969049,
            "status_code": "OK",
            "status_message": "",
            "attributes": {
                "mlflow.traceRequestId": '"tr-123"',
                "mlflow.spanType": '"LLM"',
                "mlflow.spanInputs": '{"input": 1}',
                "mlflow.spanOutputs": "2",
                "key": "3",
            },
            "events": [],
        }
    )

    assert span.request_id == "tr-123"
    assert span.name == "test"
    assert span.span_type == SpanType.LLM
    assert span.parent_id is None
    assert span.status == SpanStatus(SpanStatusCode.OK, description="")
    assert span.inputs == {"input": 1}
    assert span.outputs == 2
    assert span.events == []


def test_to_immutable_span():
    trace_id = "tr-12345"

    tracer = _get_tracer("test")
    with tracer.start_as_current_span("parent") as parent_span:
        live_span = LiveSpan(parent_span, trace_id=trace_id, span_type=SpanType.LLM)
        live_span.set_inputs({"input": 1})
        live_span.set_outputs(2)
        live_span.set_attribute("key", 3)
        live_span.set_status("OK")
        live_span.add_event(SpanEvent("test_event", timestamp=0, attributes={"foo": "bar"}))

    span = live_span.to_immutable_span()

    assert isinstance(span, Span)
    assert span.trace_id == trace_id
    assert span._trace_id == encode_trace_id(parent_span.context.trace_id)
    assert span.span_id == encode_span_id(parent_span.context.span_id)
    assert span.name == "parent"
    assert span.start_time_ns == parent_span.start_time
    assert span.end_time_ns is not None
    assert span.parent_id is None
    assert span.inputs == {"input": 1}
    assert span.outputs == 2
    assert span.get_attribute("key") == 3
    assert span.status == SpanStatus(SpanStatusCode.OK, description="")
    assert span.events == [SpanEvent("test_event", timestamp=0, attributes={"foo": "bar"})]

    with pytest.raises(AttributeError, match="set_attribute"):
        span.set_attribute("OK")


def test_from_dict_raises_when_trace_id_is_empty():
    with pytest.raises(MlflowException, match=r"Failed to create a Span object from "):
        Span.from_dict(
            {
                "name": "predict",
                "context": {
                    "trace_id": "12345",
                    "span_id": "12345",
                },
                "parent_id": None,
                "start_time": 0,
                "end_time": 1,
                "status_code": "OK",
                "status_message": "",
                "attributes": {
                    "mlflow.traceRequestId": None,
                },
                "events": [],
            }
        )


def test_set_attribute_directly_to_otel_span():
    with mlflow.start_span("test") as span:
        span._span.set_attribute("int", 1)
        span._span.set_attribute("str", "a")

    assert span.get_attribute("int") == 1
    assert span.get_attribute("str") == "a"


@pytest.fixture
def sample_otel_span_for_conversion():
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
            "mlflow.traceRequestId": "tr-12345678901234567890123456789012",
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
def test_otel_attribute_conversion(attributes):
    """Test attribute conversion with various data types."""
    from opentelemetry.proto.common.v1.common_pb2 import KeyValue

    from mlflow.tracing.utils.otlp import _decode_otel_proto_anyvalue, _set_otel_proto_anyvalue

    # Convert attributes to proto format
    proto_attrs = []
    for key, value in attributes.items():
        kv = KeyValue()
        kv.key = key
        _set_otel_proto_anyvalue(kv.value, value)
        proto_attrs.append(kv)

    # Decode back and verify
    decoded = {}
    for kv in proto_attrs:
        decoded[kv.key] = _decode_otel_proto_anyvalue(kv.value)

    assert decoded == attributes


def test_span_to_otel_proto_conversion(sample_otel_span_for_conversion):
    """Test converting MLflow Span to OTel protobuf."""
    # Create MLflow span from OTel span
    mlflow_span = Span(sample_otel_span_for_conversion)

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
    from mlflow.tracing.utils.otlp import _set_otel_proto_anyvalue

    attr1 = otel_proto.attributes.add()
    attr1.key = "mlflow.traceRequestId"
    _set_otel_proto_anyvalue(attr1.value, '{"request": "id"}')

    attr2 = otel_proto.attributes.add()
    attr2.key = "mlflow.spanType"
    _set_otel_proto_anyvalue(attr2.value, "CHAIN")

    attr3 = otel_proto.attributes.add()
    attr3.key = "custom"
    _set_otel_proto_anyvalue(attr3.value, {"nested": {"value": 123}})

    # Add event
    event = otel_proto.events.add()
    event.name = "test_event"
    event.time_unix_nano = 1500000000
    event_attr = event.attributes.add()
    event_attr.key = "event_data"
    _set_otel_proto_anyvalue(event_attr.value, "event_value")

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


def test_otel_roundtrip_conversion(sample_otel_span_for_conversion):
    """Test that conversion roundtrip preserves data."""
    # Start with OTel span -> MLflow span
    mlflow_span = Span(sample_otel_span_for_conversion)

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

    # Verify span attributes are preserved
    assert roundtrip_span.span_type == mlflow_span.span_type
    assert roundtrip_span.inputs == mlflow_span.inputs
    assert roundtrip_span.outputs == mlflow_span.outputs

    # Verify ALL attributes are preserved by iterating through them
    # Get all attribute keys from both spans
    original_attributes = mlflow_span.attributes
    roundtrip_attributes = roundtrip_span.attributes

    # Check we have the same number of attributes
    assert len(original_attributes) == len(roundtrip_attributes)

    # Check each attribute is preserved correctly
    for attr_key in original_attributes:
        assert attr_key in roundtrip_attributes, f"Attribute {attr_key} missing after roundtrip"
        original_value = original_attributes[attr_key]
        roundtrip_value = roundtrip_attributes[attr_key]
        assert original_value == roundtrip_value, (
            f"Attribute {attr_key} changed: {original_value} != {roundtrip_value}"
        )

    # Also explicitly verify specific important attributes
    # The original span has a custom_attr that should be preserved
    original_custom_attr = mlflow_span.get_attribute("custom_attr")
    roundtrip_custom_attr = roundtrip_span.get_attribute("custom_attr")
    assert original_custom_attr == roundtrip_custom_attr
    assert original_custom_attr == {"key": "value"}

    # Verify the trace request ID is preserved
    assert roundtrip_span.request_id == mlflow_span.request_id
    assert roundtrip_span.request_id == "tr-12345678901234567890123456789012"

    # Verify events
    assert len(roundtrip_span.events) == len(mlflow_span.events)
    for orig_event, rt_event in zip(mlflow_span.events, roundtrip_span.events):
        assert rt_event.name == orig_event.name
        assert rt_event.timestamp == orig_event.timestamp
        assert rt_event.attributes == orig_event.attributes
