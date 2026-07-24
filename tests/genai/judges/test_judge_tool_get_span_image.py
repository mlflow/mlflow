import base64
import json
from typing import Any

import mlflow
from mlflow.genai.judges.tools.get_span_image import (
    _ATTACHMENT_REF_RE,
    GetSpanImageTool,
    SpanImageResult,
)
from mlflow.tracing.attachments import Attachment
from mlflow.types.llm import ToolDefinition

# 1x1 PNG-ish bytes; content is opaque to the tool, which only base64-encodes it.
_IMAGE_BYTES = b"\x89PNG\r\n\x1a\n fake red png content"
_AUDIO_BYTES = b"RIFF fake wav content"


def _make_trace_with_attachments(inputs: dict[str, Any], outputs: dict[str, Any] | None = None):
    # Persist a real trace whose span carries the given attachments, then reload it
    # so the inputs/outputs hold mlflow-attachment:// refs (as autolog would produce).
    with mlflow.start_span(name="span-with-image") as span:
        span.set_inputs(inputs)
        if outputs is not None:
            span.set_outputs(outputs)
        trace_id = span.trace_id

    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id)
    return trace, trace.data.spans[0].span_id


def test_get_span_image_tool_name():
    assert GetSpanImageTool().name == "get_span_image"


def test_get_span_image_tool_get_definition():
    definition = GetSpanImageTool().get_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.function.name == "get_span_image"
    assert "mlflow-attachment://" in definition.function.description
    assert "following user message" in definition.function.description
    assert definition.function.parameters.required == ["span_id"]
    assert "attachment_index" in definition.function.parameters.properties
    assert definition.type == "function"


def test_get_span_image_invoke_success():
    trace, span_id = _make_trace_with_attachments({
        "prompt": "describe",
        "image": Attachment(content_type="image/png", content_bytes=_IMAGE_BYTES),
    })

    result = GetSpanImageTool().invoke(trace, span_id)

    assert isinstance(result, SpanImageResult)
    assert result.span_id == span_id
    assert result.content_type == "image/png"
    assert result.data_url.startswith("data:image/png;base64,")

    encoded = result.data_url.split(",", 1)[1]
    assert base64.b64decode(encoded) == _IMAGE_BYTES


def test_get_span_image_invoke_selects_attachment_index():
    first = b"first image bytes"
    second = b"second image bytes"
    trace, span_id = _make_trace_with_attachments({
        "a": Attachment(content_type="image/png", content_bytes=first),
        "b": Attachment(content_type="image/png", content_bytes=second),
    })

    # Both indices must resolve to a distinct, correctly-decoded image.
    r0 = GetSpanImageTool().invoke(trace, span_id, attachment_index=0)
    r1 = GetSpanImageTool().invoke(trace, span_id, attachment_index=1)

    assert isinstance(r0, SpanImageResult)
    assert isinstance(r1, SpanImageResult)
    decoded = {base64.b64decode(r.data_url.split(",", 1)[1]) for r in (r0, r1)}
    assert decoded == {first, second}


def test_get_span_image_invoke_span_not_found():
    trace, _ = _make_trace_with_attachments({
        "image": Attachment(content_type="image/png", content_bytes=_IMAGE_BYTES)
    })

    result = GetSpanImageTool().invoke(trace, "nonexistent-span")

    assert isinstance(result, str)
    assert "not found" in result


def test_get_span_image_invoke_no_attachment():
    trace, span_id = _make_trace_with_attachments({"prompt": "no image here"})

    result = GetSpanImageTool().invoke(trace, span_id)

    assert isinstance(result, str)
    assert "no mlflow-attachment" in result


def test_get_span_image_invoke_default_skips_non_image_before_image():
    # A non-image attachment sitting before an image one must not shadow the image on
    # the default (no explicit index) call.
    trace, span_id = _make_trace_with_attachments({
        "audio": Attachment(content_type="audio/wav", content_bytes=_AUDIO_BYTES),
        "image": Attachment(content_type="image/png", content_bytes=_IMAGE_BYTES),
    })

    result = GetSpanImageTool().invoke(trace, span_id)

    assert isinstance(result, SpanImageResult)
    assert result.content_type == "image/png"
    assert base64.b64decode(result.data_url.split(",", 1)[1]) == _IMAGE_BYTES


def test_get_span_image_invoke_default_no_image_among_non_images():
    trace, span_id = _make_trace_with_attachments({
        "audio": Attachment(content_type="audio/wav", content_bytes=_AUDIO_BYTES)
    })

    result = GetSpanImageTool().invoke(trace, span_id)

    assert isinstance(result, str)
    assert "no image attachment found" in result


def test_get_span_image_invoke_explicit_index_at_non_image_errors():
    # An explicit index pointing at a non-image ref still reports the non-image error,
    # even when an image ref exists elsewhere in the span.
    trace, span_id = _make_trace_with_attachments({
        "audio": Attachment(content_type="audio/wav", content_bytes=_AUDIO_BYTES),
        "image": Attachment(content_type="image/png", content_bytes=_IMAGE_BYTES),
    })

    # Find the index of the non-image ref so the assertion is order-independent.
    span = next(s for s in trace.data.spans if s.span_id == span_id)
    refs = _ATTACHMENT_REF_RE.findall(json.dumps(span.to_dict(), default=str))
    audio_index = next(
        i for i, ref in enumerate(refs) if Attachment.parse_ref(ref)["content_type"] == "audio/wav"
    )

    result = GetSpanImageTool().invoke(trace, span_id, attachment_index=audio_index)

    assert isinstance(result, str)
    assert "not an image" in result
    assert "audio/wav" in result


def test_get_span_image_invoke_attachment_index_out_of_range():
    trace, span_id = _make_trace_with_attachments({
        "image": Attachment(content_type="image/png", content_bytes=_IMAGE_BYTES)
    })

    result = GetSpanImageTool().invoke(trace, span_id, attachment_index=5)

    assert isinstance(result, str)
    assert "out of range" in result
