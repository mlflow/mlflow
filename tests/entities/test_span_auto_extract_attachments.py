import base64

from mlflow.entities.span import LiveSpan
from mlflow.tracing.attachments import Attachment


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
