import base64
import logging
from contextlib import contextmanager

import pytest
from pydantic import BaseModel

import mlflow.entities.span as span_module
from mlflow.entities.span import LiveSpan
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.tracing.attachments import Attachment
from mlflow.tracing.trace_manager import InMemoryTraceManager


def _make_live_span(trace_id="tr-test123"):
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")
    otel_span = tracer.start_span("test_span")
    return LiveSpan(otel_span, trace_id=trace_id)


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)
PNG_DATA_URI = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"

WAV_B64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEfmt ").decode()


# --- Data URI extraction ---


def test_extracts_image_data_uri():
    span = _make_live_span()
    span.set_inputs({"image": PNG_DATA_URI})

    inputs = span.inputs
    assert inputs["image"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 1
    att = next(iter(span._attachments.values()))
    assert att.content_type == "image/png"
    assert att.content_bytes == PNG_BYTES


def test_extracts_nested_data_uri():
    span = _make_live_span()
    span.set_inputs({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": PNG_DATA_URI},
                    },
                ],
            }
        ]
    })

    inputs = span.inputs
    url = inputs["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("mlflow-attachment://")
    assert len(span._attachments) == 1


def test_leaves_http_urls_alone():
    span = _make_live_span()
    url = "https://example.com/photo.png"
    span.set_inputs({"image_url": {"url": url}})
    assert span.inputs["image_url"]["url"] == url
    assert len(span._attachments) == 0


def test_leaves_plain_strings_alone():
    span = _make_live_span()
    span.set_inputs({"text": "hello world"})
    assert span.inputs["text"] == "hello world"
    assert len(span._attachments) == 0


def test_handles_invalid_base64_gracefully():
    span = _make_live_span()
    bad_uri = "data:image/png;base64,!!!not-valid-base64!!!"
    span.set_inputs({"image": bad_uri})
    assert span.inputs["image"] == bad_uri
    assert len(span._attachments) == 0


def test_rejects_base64_with_trailing_garbage():
    # "Zg==!!!" is silently accepted by b64decode without validate=True
    span = _make_live_span()
    bad_uri = "data:image/png;base64,Zg==!!!"
    span.set_inputs({"image": bad_uri})
    assert span.inputs["image"] == bad_uri
    assert len(span._attachments) == 0


def test_handles_empty_mime_type():
    span = _make_live_span()
    bad_uri = "data:;base64,dGVzdA=="
    span.set_inputs({"val": bad_uri})
    assert span.inputs["val"] == bad_uri
    assert len(span._attachments) == 0


def test_multiple_data_uris():
    span = _make_live_span()
    span.set_inputs({"img1": PNG_DATA_URI, "img2": PNG_DATA_URI})
    assert span.inputs["img1"].startswith("mlflow-attachment://")
    assert span.inputs["img2"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 2


# --- Structured content extraction ---


def test_extracts_input_audio():
    span = _make_live_span()
    span.set_inputs({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What does this say?"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": WAV_B64, "format": "wav"},
                    },
                ],
            }
        ]
    })

    audio_part = span.inputs["messages"][0]["content"][1]
    assert audio_part["type"] == "input_audio"
    assert audio_part["input_audio"]["data"].startswith("mlflow-attachment://")
    assert audio_part["input_audio"]["format"] == "wav"
    assert len(span._attachments) == 1
    att = next(iter(span._attachments.values()))
    assert att.content_type == "audio/wav"


def test_extracts_b64_json():
    span = _make_live_span()
    img_b64 = base64.b64encode(PNG_BYTES).decode()
    span.set_outputs({"data": [{"b64_json": img_b64, "revised_prompt": "a sunset"}]})

    output = span.outputs
    item = output["data"][0]
    assert item["b64_json"].startswith("mlflow-attachment://")
    assert item["revised_prompt"] == "a sunset"
    assert len(span._attachments) == 1
    att = next(iter(span._attachments.values()))
    assert att.content_type == "image/png"
    assert att.content_bytes == PNG_BYTES


def test_input_audio_with_invalid_base64():
    span = _make_live_span()
    span.set_inputs({
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": "!!!bad!!!", "format": "wav"},
            }
        ]
    })
    audio_part = span.inputs["content"][0]
    assert audio_part["input_audio"]["data"] == "!!!bad!!!"
    assert len(span._attachments) == 0


def test_structured_content_with_sibling_data_uri():
    span = _make_live_span()
    span.set_inputs({
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": WAV_B64, "format": "wav"},
                "extra_image": PNG_DATA_URI,
            }
        ]
    })
    part = span.inputs["content"][0]
    assert part["input_audio"]["data"].startswith("mlflow-attachment://")
    assert part["extra_image"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 2


def test_mixed_content_parts():
    span = _make_live_span()
    span.set_inputs({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe both"},
                    {
                        "type": "image_url",
                        "image_url": {"url": PNG_DATA_URI},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": WAV_B64, "format": "mp3"},
                    },
                ],
            }
        ]
    })
    content = span.inputs["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe both"}
    assert content[1]["image_url"]["url"].startswith("mlflow-attachment://")
    assert content[2]["input_audio"]["data"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 2


# --- Anthropic image pattern ---


def test_extracts_anthropic_image():
    span = _make_live_span()
    span.set_inputs({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(PNG_BYTES).decode(),
                        },
                    },
                ],
            }
        ]
    })

    content = span.inputs["messages"][0]["content"]
    assert content[1]["type"] == "image"
    assert content[1]["source"]["data"].startswith("mlflow-attachment://")
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/png"
    assert len(span._attachments) == 1
    att = next(iter(span._attachments.values()))
    assert att.content_type == "image/png"
    assert att.content_bytes == PNG_BYTES


def test_extracts_multiple_anthropic_images():
    span = _make_live_span()
    img2_bytes = b"fake jpeg bytes"
    span.set_inputs({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(PNG_BYTES).decode(),
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(img2_bytes).decode(),
                        },
                    },
                ],
            }
        ]
    })

    content = span.inputs["messages"][0]["content"]
    assert content[0]["source"]["data"].startswith("mlflow-attachment://")
    assert content[1]["source"]["data"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 2


# --- Audio output pattern ---


def test_extracts_audio_output():
    span = _make_live_span()
    audio_b64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEfmt ").decode()
    span.set_outputs({
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "audio": {
                        "id": "audio_123",
                        "data": audio_b64,
                        "transcript": "Hello world",
                    },
                }
            }
        ]
    })

    outputs = span.outputs
    audio = outputs["choices"][0]["message"]["audio"]
    assert audio["data"].startswith("mlflow-attachment://")
    assert audio["transcript"] == "Hello world"
    assert audio["id"] == "audio_123"
    assert len(span._attachments) == 1
    att = next(iter(span._attachments.values()))
    assert att.content_type == "audio/wav"


def test_extracts_b64_json_multiple():
    span = _make_live_span()
    img_b64 = base64.b64encode(PNG_BYTES).decode()
    span.set_outputs({
        "data": [
            {"b64_json": img_b64, "revised_prompt": "a circle"},
            {"b64_json": img_b64, "revised_prompt": "a triangle"},
        ]
    })

    output = span.outputs
    assert output["data"][0]["b64_json"].startswith("mlflow-attachment://")
    assert output["data"][1]["b64_json"].startswith("mlflow-attachment://")
    assert output["data"][0]["revised_prompt"] == "a circle"
    assert output["data"][1]["revised_prompt"] == "a triangle"
    assert len(span._attachments) == 2


# --- Bedrock image pattern ---


def test_extracts_bedrock_image():
    span = _make_live_span()
    img_b64 = base64.b64encode(PNG_BYTES).decode()
    span.set_outputs({
        "output": {
            "message": {
                "content": [
                    {"text": "Here is the image."},
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": img_b64},
                        }
                    },
                ]
            }
        }
    })

    content = span.outputs["output"]["message"]["content"]
    assert content[0] == {"text": "Here is the image."}
    img_block = content[1]
    assert img_block["image"]["format"] == "png"
    assert img_block["image"]["source"]["bytes"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 1
    att = next(iter(span._attachments.values()))
    assert att.content_type == "image/png"
    assert att.content_bytes == PNG_BYTES


def test_bedrock_image_with_invalid_base64():
    span = _make_live_span()
    span.set_outputs({
        "content": [
            {
                "image": {
                    "format": "png",
                    "source": {"bytes": "!!!bad!!!"},
                }
            }
        ]
    })
    img_block = span.outputs["content"][0]
    assert img_block["image"]["source"]["bytes"] == "!!!bad!!!"
    assert len(span._attachments) == 0


# --- Gemini inline_data pattern ---


def test_extracts_gemini_inline_data():
    span = _make_live_span()
    img_b64 = base64.b64encode(PNG_BYTES).decode()
    span.set_outputs({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Here is what I see."},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img_b64,
                            }
                        },
                    ]
                }
            }
        ]
    })

    parts = span.outputs["candidates"][0]["content"]["parts"]
    assert parts[0] == {"text": "Here is what I see."}
    inline = parts[1]
    assert inline["inline_data"]["mime_type"] == "image/png"
    assert inline["inline_data"]["data"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 1
    att = next(iter(span._attachments.values()))
    assert att.content_type == "image/png"
    assert att.content_bytes == PNG_BYTES


def test_extracts_gemini_inline_data_bytes_repr():
    # Gemini SDK Pydantic serialization produces repr(bytes) instead of base64
    span = _make_live_span()
    bytes_repr = repr(PNG_BYTES)
    span.set_outputs({
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "A small image."},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": bytes_repr,
                            }
                        },
                    ]
                }
            }
        ]
    })

    parts = span.outputs["candidates"][0]["content"]["parts"]
    assert parts[0] == {"text": "A small image."}
    inline = parts[1]
    assert inline["inline_data"]["data"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 1
    att = next(iter(span._attachments.values()))
    assert att.content_type == "image/png"
    assert att.content_bytes == PNG_BYTES


def test_gemini_inline_data_with_invalid_base64():
    span = _make_live_span()
    span.set_outputs({
        "parts": [
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": "!!!bad!!!",
                }
            }
        ]
    })
    inline = span.outputs["parts"][0]
    assert inline["inline_data"]["data"] == "!!!bad!!!"
    assert len(span._attachments) == 0


# --- Responses API image_generation_call pattern ---


def test_extracts_responses_api_image_generation():
    span = _make_live_span()
    img_b64 = base64.b64encode(PNG_BYTES).decode()
    span.set_outputs({
        "output": [
            {
                "type": "image_generation_call",
                "result": img_b64,
                "output_format": "png",
                "revised_prompt": "a blue square",
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Here is the image."}],
            },
        ]
    })

    outputs = span.outputs
    img_call = outputs["output"][0]
    assert img_call["type"] == "image_generation_call"
    assert img_call["result"].startswith("mlflow-attachment://")
    assert img_call["revised_prompt"] == "a blue square"
    assert img_call["output_format"] == "png"
    msg = outputs["output"][1]
    assert msg["type"] == "message"
    assert len(span._attachments) == 1
    att = next(iter(span._attachments.values()))
    assert att.content_type == "image/png"
    assert att.content_bytes == PNG_BYTES


def test_responses_api_image_generation_with_invalid_base64():
    span = _make_live_span()
    span.set_outputs({
        "output": [
            {
                "type": "image_generation_call",
                "result": "!!!bad!!!",
                "output_format": "png",
            }
        ]
    })
    img_call = span.outputs["output"][0]
    assert img_call["result"] == "!!!bad!!!"
    assert len(span._attachments) == 0


# --- Two-pass serialization extraction ---


def test_extracts_base64_from_pydantic_model():
    """Pydantic models aren't traversable in the first pass but become
    plain dicts after JSON serialization. The second pass should extract
    the base64 data from the serialized form.
    """

    class AudioOutput(BaseModel):
        transcript: str
        audio: dict[str, str]

    audio_b64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVEfmt ").decode()
    output = AudioOutput(
        transcript="Hello",
        audio={"data": audio_b64, "id": "audio_123"},
    )

    span = _make_live_span()
    span.set_outputs({"result": output})

    outputs = span.outputs
    assert outputs["result"]["audio"]["data"].startswith("mlflow-attachment://")
    assert outputs["result"]["transcript"] == "Hello"
    assert len(span._attachments) == 1
    att = next(iter(span._attachments.values()))
    assert att.content_type == "audio/wav"


def test_two_pass_with_explicit_attachment_and_pydantic():
    """When a span has both an explicit Attachment (first pass) AND a Pydantic
    model with base64 (second pass), both should be extracted.
    """

    class ImageResult(BaseModel):
        b64_json: str
        revised_prompt: str

    img_b64 = base64.b64encode(PNG_BYTES).decode()
    pydantic_output = ImageResult(b64_json=img_b64, revised_prompt="a sunset")
    explicit_att = Attachment(content_type="image/png", content_bytes=PNG_BYTES)

    span = _make_live_span()
    span.set_outputs({"image": pydantic_output, "thumbnail": explicit_att})

    outputs = span.outputs
    assert outputs["thumbnail"].startswith("mlflow-attachment://")
    assert outputs["image"]["b64_json"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 2


# --- Unity Catalog table traces (no artifact store) ---

UC_TRACE_ID = "trace:/cat.sch.mlflow_experiment/abc123"
_NEXT_OTEL_TRACE_ID = 910_000


@contextmanager
def _registered_trace(trace_info: TraceInfo):
    global _NEXT_OTEL_TRACE_ID
    _NEXT_OTEL_TRACE_ID += 1
    otel_trace_id = _NEXT_OTEL_TRACE_ID
    manager = InMemoryTraceManager.get_instance()
    manager.register_trace(otel_trace_id, trace_info)
    try:
        yield
    finally:
        manager.pop_trace(otel_trace_id)


def _trace_info(trace_id: str, trace_location: TraceLocation) -> TraceInfo:
    return TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location,
        request_time=0,
        state=TraceState.OK,
    )


def test_uc_table_trace_keeps_openai_image_data_uri_inline(caplog):
    span_module._logged_uc_inline_attachment_notice = False
    span = _make_live_span(trace_id=UC_TRACE_ID)
    with caplog.at_level(logging.INFO, logger="mlflow.entities.span"):
        span.set_inputs({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is on this image?"},
                        {"type": "image_url", "image_url": {"url": PNG_DATA_URI}},
                    ],
                }
            ]
        })
        span.set_outputs({"image": PNG_DATA_URI})

    url = span.inputs["messages"][0]["content"][1]["image_url"]["url"]
    assert url == PNG_DATA_URI
    assert span.outputs["image"] == PNG_DATA_URI
    assert span._attachments == {}
    notices = [
        record
        for record in caplog.records
        if "not supported for Unity Catalog table traces" in record.message
    ]
    assert len(notices) == 1
    assert notices[0].levelno == logging.INFO
    assert "Trace data is corrupted" not in caplog.text


@pytest.mark.parametrize(
    "trace_location",
    [
        TraceLocation.from_databricks_uc_table_prefix("cat", "sch", "mlflow_experiment"),
        TraceLocation.from_databricks_uc_schema("cat", "sch"),
    ],
)
def test_registered_uc_location_keeps_data_uri_inline(trace_location):
    # The trace id is intentionally not v4. The registered location decides.
    info = _trace_info("tr-uc-location", trace_location)
    with _registered_trace(info):
        span = _make_live_span(trace_id=info.trace_id)
        span.set_inputs({"image": PNG_DATA_URI})

    assert span.inputs["image"] == PNG_DATA_URI
    assert span._attachments == {}


def test_registered_experiment_trace_still_extracts_data_uri():
    info = _trace_info("tr-experiment", TraceLocation.from_experiment_id("0"))
    with _registered_trace(info):
        span = _make_live_span(trace_id=info.trace_id)
        span.set_inputs({"image": PNG_DATA_URI})

    assert span.inputs["image"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 1


def test_v4_trace_id_with_experiment_location_still_extracts():
    info = _trace_info("trace:/0/abc123", TraceLocation.from_experiment_id("0"))
    with _registered_trace(info):
        span = _make_live_span(trace_id=info.trace_id)
        span.set_inputs({"image": PNG_DATA_URI})

    assert span.inputs["image"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 1


def test_uc_table_trace_leaves_structured_base64_inline():
    span = _make_live_span(trace_id=UC_TRACE_ID)
    span.set_inputs({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "wav"}},
                ],
            }
        ]
    })

    audio = span.inputs["messages"][0]["content"][0]["input_audio"]
    assert audio["data"] == WAV_B64
    assert span._attachments == {}


def test_uc_table_trace_inlines_explicit_attachment():
    span = _make_live_span(trace_id=UC_TRACE_ID)
    span.set_inputs({"image": Attachment(content_type="image/png", content_bytes=PNG_BYTES)})

    image = span.inputs["image"]
    assert image.startswith("data:image/png;base64,")
    assert base64.b64decode(image.split(",", 1)[1]) == PNG_BYTES
    assert span._attachments == {}


def test_uc_table_trace_inlines_explicit_attachment_when_extraction_disabled(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_EXTRACT_ATTACHMENTS", "false")
    span = _make_live_span(trace_id=UC_TRACE_ID)
    span.set_inputs({"image": Attachment(content_type="image/png", content_bytes=PNG_BYTES)})

    assert span.inputs["image"].startswith("data:image/png;base64,")
    assert span._attachments == {}


def test_uc_table_trace_discards_explicit_attachment_over_size_limit(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_MAX_ATTACHMENT_SIZE", "1")
    span = _make_live_span(trace_id=UC_TRACE_ID)
    span.set_inputs({"image": Attachment(content_type="image/png", content_bytes=PNG_BYTES)})

    assert "[Attachment too large:" in span.inputs["image"]
    assert span._attachments == {}


# --- Opt-out ---


def test_opt_out_via_env_var(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_EXTRACT_ATTACHMENTS", "false")
    span = _make_live_span()
    span.set_inputs({"image": PNG_DATA_URI})
    assert span.inputs["image"] == PNG_DATA_URI
    assert len(span._attachments) == 0


def test_explicit_attachment_still_works_when_opted_out(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_EXTRACT_ATTACHMENTS", "false")
    span = _make_live_span()
    att = Attachment(content_type="image/png", content_bytes=PNG_BYTES)
    span.set_inputs({"image": att})
    assert span.inputs["image"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 1


# --- Attachment size limit ---


def test_attachment_under_size_limit_is_extracted(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_MAX_ATTACHMENT_SIZE", str(len(PNG_BYTES) + 1))
    span = _make_live_span()
    span.set_inputs({"image": PNG_DATA_URI})

    assert span.inputs["image"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 1


def test_attachment_over_size_limit_is_discarded(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_MAX_ATTACHMENT_SIZE", str(len(PNG_BYTES) - 1))
    span = _make_live_span()
    span.set_inputs({"image": PNG_DATA_URI})

    assert "[Attachment too large:" in span.inputs["image"]
    assert len(span._attachments) == 0


def test_attachment_size_limit_unset_allows_all():
    # Default is None (unset) — no limit enforced
    span = _make_live_span()
    span.set_inputs({"image": PNG_DATA_URI})

    assert span.inputs["image"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 1


def test_structured_content_over_size_limit_is_discarded(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_MAX_ATTACHMENT_SIZE", "1")
    span = _make_live_span()
    span.set_inputs({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": WAV_B64, "format": "wav"},
                    }
                ],
            }
        ]
    })

    audio_part = span.inputs["messages"][0]["content"][0]
    assert "[Attachment too large:" in audio_part["input_audio"]["data"]
    assert len(span._attachments) == 0


def test_explicit_attachment_over_size_limit_is_discarded(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_MAX_ATTACHMENT_SIZE", "1")
    span = _make_live_span()
    att = Attachment(content_type="image/png", content_bytes=PNG_BYTES)
    span.set_inputs({"image": att})

    assert "[Attachment too large:" in span.inputs["image"]
    assert len(span._attachments) == 0


def test_attachment_size_limit_negative_treated_as_disabled(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_MAX_ATTACHMENT_SIZE", "-1")
    span = _make_live_span()
    span.set_inputs({"image": PNG_DATA_URI})

    assert span.inputs["image"].startswith("mlflow-attachment://")
    assert len(span._attachments) == 1
