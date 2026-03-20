from mlflow.entities.span import LiveSpan, Span
from mlflow.tracing.attachments import Attachment


def _make_live_span(trace_id="tr-test123"):
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")
    otel_span = tracer.start_span("test_span")
    return LiveSpan(otel_span, trace_id=trace_id)


def test_set_inputs_extracts_attachment():
    span = _make_live_span()
    att = Attachment(content_type="image/png", content_bytes=b"img")
    span.set_inputs({"image": att})

    assert att.id in span._attachments
    assert span._attachments[att.id] is att
    inputs = span.inputs
    assert isinstance(inputs["image"], str)
    assert inputs["image"].startswith("mlflow-attachment://")
    parsed = Attachment.parse_ref(inputs["image"])
    assert parsed["trace_id"] == "tr-test123"
    assert parsed["attachment_id"] == att.id


def test_set_outputs_extracts_attachment():
    span = _make_live_span()
    att = Attachment(content_type="audio/wav", content_bytes=b"audio")
    span.set_outputs({"sound": att})

    assert att.id in span._attachments
    outputs = span.outputs
    assert isinstance(outputs["sound"], str)
    parsed = Attachment.parse_ref(outputs["sound"])
    assert parsed["content_type"] == "audio/wav"


def test_nested_dict_extraction():
    span = _make_live_span()
    att = Attachment(content_type="image/jpeg", content_bytes=b"jpg")
    span.set_inputs({"nested": {"deep": att}})

    assert att.id in span._attachments
    inputs = span.inputs
    assert isinstance(inputs["nested"]["deep"], str)
    assert att.id in inputs["nested"]["deep"]


def test_list_extraction():
    span = _make_live_span()
    att1 = Attachment(content_type="image/png", content_bytes=b"a")
    att2 = Attachment(content_type="image/png", content_bytes=b"b")
    span.set_inputs({"images": [att1, att2]})

    assert att1.id in span._attachments
    assert att2.id in span._attachments
    inputs = span.inputs
    assert len(inputs["images"]) == 2
    assert all(isinstance(v, str) for v in inputs["images"])


def test_mixed_values_only_replaces_attachments():
    span = _make_live_span()
    att = Attachment(content_type="image/png", content_bytes=b"img")
    span.set_inputs({"image": att, "text": "hello", "number": 42})

    assert att.id in span._attachments
    inputs = span.inputs
    assert inputs["text"] == "hello"
    assert inputs["number"] == 42
    assert isinstance(inputs["image"], str)


def test_non_dict_input_without_attachment():
    span = _make_live_span()
    span.set_inputs("plain string")
    assert span._attachments == {}
    assert span.inputs == "plain string"


def test_tuple_extraction():
    span = _make_live_span()
    att = Attachment(content_type="image/png", content_bytes=b"img")
    span.set_inputs({"images": (att, "text")})

    assert att.id in span._attachments
    inputs = span.inputs
    assert isinstance(inputs["images"], list)
    assert isinstance(inputs["images"][0], str)
    assert inputs["images"][0].startswith("mlflow-attachment://")
    assert inputs["images"][1] == "text"


def test_set_inputs_twice_accumulates_attachments():
    span = _make_live_span()
    att1 = Attachment(content_type="image/png", content_bytes=b"first")
    att2 = Attachment(content_type="image/jpeg", content_bytes=b"second")

    span.set_inputs({"img": att1})
    span.set_outputs({"img": att2})

    assert att1.id in span._attachments
    assert att2.id in span._attachments
    assert len(span._attachments) == 2


def test_to_immutable_span_propagates_attachments():
    span = _make_live_span()
    att = Attachment(content_type="image/png", content_bytes=b"img")
    span.set_inputs({"image": att})

    immutable = span.to_immutable_span()
    assert isinstance(immutable, Span)
    assert att.id in immutable._attachments
    assert immutable._attachments[att.id] is att
