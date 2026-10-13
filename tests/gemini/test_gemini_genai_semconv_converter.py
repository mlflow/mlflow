import json
from unittest.mock import patch

import pytest
from google import genai

import mlflow
from mlflow.gemini.genai_semconv_converter import _convert_part
from mlflow.tracing.constant import GenAiSemconvKey

from tests.gemini.test_gemini_autolog import (
    _dummy_generate_content,
    _generate_content_response,
    multiply,
)
from tests.tracing.helper import capture_otel_export, reset_autolog_state  # noqa: F401

MODEL = "gemini-1.5-flash"


@pytest.fixture(autouse=True)
def enable_genai_semconv(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_OTEL_GENAI_SEMCONV", "true")
    return


def _get_llm_span(exporter, processor):
    processor.force_flush(timeout_millis=5000)
    spans = exporter.get_finished_spans()
    return next(
        s for s in spans if s.attributes.get(GenAiSemconvKey.OPERATION_NAME) == "generate_content"
    )


@pytest.mark.usefixtures("reset_autolog_state")
def test_autolog_basic(capture_otel_export):
    exporter, processor = capture_otel_export

    mlflow.gemini.autolog()
    with patch(
        "google.genai.models.Models._generate_content",
        new=_dummy_generate_content(is_async=False),
    ):
        client = genai.Client(api_key="dummy")
        client.models.generate_content(model=MODEL, contents="test content")

    llm_span = _get_llm_span(exporter, processor)
    assert llm_span.attributes[GenAiSemconvKey.OPERATION_NAME] == "generate_content"
    assert llm_span.attributes[GenAiSemconvKey.REQUEST_MODEL] == MODEL

    input_msgs = json.loads(llm_span.attributes[GenAiSemconvKey.INPUT_MESSAGES])
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["type"] == "text"
    assert input_msgs[0]["parts"][0]["content"] == "test content"

    output_msgs = json.loads(llm_span.attributes[GenAiSemconvKey.OUTPUT_MESSAGES])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    assert output_msgs[0]["parts"][0]["content"] == "test answer"
    assert not any(k.startswith("mlflow.") for k in llm_span.attributes)


@pytest.mark.usefixtures("reset_autolog_state")
def test_autolog_with_tool_calls(capture_otel_export):
    exporter, processor = capture_otel_export

    tool_call_content = {
        "parts": [
            {
                "function_call": {
                    "name": "multiply",
                    "args": {"a": 57.0, "b": 44.0},
                }
            }
        ],
        "role": "model",
    }
    response = _generate_content_response(tool_call_content)

    def _generate_content(self, model, contents, config):
        return response

    mlflow.gemini.autolog()
    with patch("google.genai.models.Models._generate_content", new=_generate_content):
        client = genai.Client(api_key="dummy")
        client.models.generate_content(
            model=MODEL,
            contents="How much is 57 * 44?",
            config=genai.types.GenerateContentConfig(
                tools=[multiply],
                automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )

    llm_span = _get_llm_span(exporter, processor)
    assert llm_span.attributes[GenAiSemconvKey.OPERATION_NAME] == "generate_content"
    assert llm_span.attributes[GenAiSemconvKey.REQUEST_MODEL] == MODEL

    input_msgs = json.loads(llm_span.attributes[GenAiSemconvKey.INPUT_MESSAGES])
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["content"] == "How much is 57 * 44?"

    output_msgs = json.loads(llm_span.attributes[GenAiSemconvKey.OUTPUT_MESSAGES])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    tool_part = output_msgs[0]["parts"][0]
    assert tool_part["type"] == "tool_call"
    assert tool_part["name"] == "multiply"
    assert tool_part["arguments"] == {"a": 57.0, "b": 44.0}

    assert GenAiSemconvKey.TOOL_DEFINITIONS in llm_span.attributes
    assert "multiply" in llm_span.attributes[GenAiSemconvKey.TOOL_DEFINITIONS]
    assert not any(k.startswith("mlflow.") for k in llm_span.attributes)


@pytest.mark.parametrize(
    ("part", "expected"),
    [
        # inline_data (image)
        (
            {"inline_data": {"data": "iVBOR...", "mime_type": "image/png"}},
            {
                "type": "blob",
                "modality": "image",
                "mime_type": "image/png",
                "content": "iVBOR...",
            },
        ),
        # file_data (image)
        (
            {"file_data": {"file_uri": "gs://bucket/img.jpg", "mime_type": "image/jpeg"}},
            {
                "type": "uri",
                "modality": "image",
                "mime_type": "image/jpeg",
                "uri": "gs://bucket/img.jpg",
            },
        ),
        # inline_data (audio)
        (
            {"inline_data": {"data": "audiodata", "mime_type": "audio/mp3"}},
            {
                "type": "blob",
                "modality": "audio",
                "mime_type": "audio/mp3",
                "content": "audiodata",
            },
        ),
        # file_data (video)
        (
            {"file_data": {"file_uri": "gs://bucket/vid.mp4", "mime_type": "video/mp4"}},
            {
                "type": "uri",
                "modality": "video",
                "mime_type": "video/mp4",
                "uri": "gs://bucket/vid.mp4",
            },
        ),
    ],
)
def test_convert_part_multimodal(part, expected):
    assert _convert_part(part) == expected
