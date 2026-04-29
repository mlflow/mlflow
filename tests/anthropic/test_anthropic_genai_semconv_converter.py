import json
from unittest.mock import patch

import anthropic
import pytest
from packaging.version import Version

import mlflow
from mlflow.anthropic.genai_semconv_converter import _convert_block
from mlflow.tracing.constant import GenAiSemconvKey

from tests.anthropic.test_anthropic_autolog import (
    DUMMY_CREATE_MESSAGE_REQUEST,
    DUMMY_CREATE_MESSAGE_RESPONSE,
    DUMMY_CREATE_MESSAGE_WITH_TOOLS_REQUEST,
    DUMMY_CREATE_MESSAGE_WITH_TOOLS_RESPONSE,
)
from tests.tracing.helper import capture_otel_export, reset_autolog_state  # noqa: F401

if Version(anthropic.__version__) < Version("0.55"):
    pytest.skip("GenAI semconv converter requires anthropic >= 0.55", allow_module_level=True)


@pytest.fixture(autouse=True)
def enable_genai_semconv(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_OTEL_GENAI_SEMCONV", "true")
    return


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    return anthropic.Anthropic(api_key="test")


def _get_chat_span(exporter, processor):
    processor.force_flush(timeout_millis=5000)
    spans = exporter.get_finished_spans()
    return next(s for s in spans if s.attributes.get(GenAiSemconvKey.OPERATION_NAME) == "chat")


@pytest.mark.usefixtures("reset_autolog_state")
def test_autolog_basic(client, capture_otel_export):
    exporter, processor = capture_otel_export

    mlflow.anthropic.autolog()
    with patch(
        "anthropic._base_client.SyncAPIClient.post",
        return_value=DUMMY_CREATE_MESSAGE_RESPONSE,
    ) as mock_post:
        client.messages.create(**DUMMY_CREATE_MESSAGE_REQUEST)
        mock_post.assert_called_once()

    chat_span = _get_chat_span(exporter, processor)
    assert chat_span.attributes[GenAiSemconvKey.OPERATION_NAME] == "chat"
    assert chat_span.attributes[GenAiSemconvKey.REQUEST_MODEL] == "test_model"
    assert chat_span.attributes[GenAiSemconvKey.REQUEST_MAX_TOKENS] == 1024

    input_msgs = json.loads(chat_span.attributes[GenAiSemconvKey.INPUT_MESSAGES])
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["type"] == "text"
    assert input_msgs[0]["parts"][0]["content"] == "test message"

    output_msgs = json.loads(chat_span.attributes[GenAiSemconvKey.OUTPUT_MESSAGES])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    assert output_msgs[0]["parts"][0]["content"] == "test answer"

    assert chat_span.attributes[GenAiSemconvKey.RESPONSE_MODEL] == "test_model"
    assert chat_span.attributes[GenAiSemconvKey.RESPONSE_ID] == "test_id"
    assert not any(k.startswith("mlflow.") for k in chat_span.attributes)


@pytest.mark.usefixtures("reset_autolog_state")
def test_autolog_with_tool_calls(client, capture_otel_export):
    exporter, processor = capture_otel_export

    mlflow.anthropic.autolog()
    with patch(
        "anthropic._base_client.SyncAPIClient.post",
        return_value=DUMMY_CREATE_MESSAGE_WITH_TOOLS_RESPONSE,
    ) as mock_post:
        client.messages.create(**DUMMY_CREATE_MESSAGE_WITH_TOOLS_REQUEST)
        mock_post.assert_called_once()

    chat_span = _get_chat_span(exporter, processor)
    assert chat_span.attributes[GenAiSemconvKey.OPERATION_NAME] == "chat"
    assert chat_span.attributes[GenAiSemconvKey.REQUEST_MODEL] == "test_model"

    tool_defs = json.loads(chat_span.attributes[GenAiSemconvKey.TOOL_DEFINITIONS])
    assert tool_defs[0]["name"] == "get_unit"
    assert tool_defs[1]["name"] == "get_weather"

    input_msgs = json.loads(chat_span.attributes[GenAiSemconvKey.INPUT_MESSAGES])
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["content"] == "What's the weather like in San Francisco?"
    # Assistant message with text + tool call
    assert input_msgs[1]["role"] == "assistant"
    assert input_msgs[1]["parts"][0]["type"] == "text"
    assert input_msgs[1]["parts"][1]["type"] == "tool_call"
    assert input_msgs[1]["parts"][1]["id"] == "tool_123"
    assert input_msgs[1]["parts"][1]["name"] == "get_unit"
    assert input_msgs[1]["parts"][1]["arguments"] == {"location": "San Francisco"}
    # Tool result
    assert input_msgs[2]["role"] == "tool"
    assert input_msgs[2]["parts"][0]["type"] == "tool_call_response"
    assert input_msgs[2]["parts"][0]["id"] == "tool_123"
    assert input_msgs[2]["parts"][0]["result"] == "celsius"

    output_msgs = json.loads(chat_span.attributes[GenAiSemconvKey.OUTPUT_MESSAGES])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    tool_part = next(p for p in output_msgs[0]["parts"] if p["type"] == "tool_call")
    assert tool_part["name"] == "get_weather"

    assert chat_span.attributes[GenAiSemconvKey.RESPONSE_MODEL] == "test_model"
    assert not any(k.startswith("mlflow.") for k in chat_span.attributes)


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        # Image base64
        (
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBOR...",
                },
            },
            {
                "type": "blob",
                "modality": "image",
                "mime_type": "image/png",
                "content": "iVBOR...",
            },
        ),
        # Image URL
        (
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/img.png"},
            },
            {"type": "uri", "modality": "image", "uri": "https://example.com/img.png"},
        ),
        # Document base64
        (
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi...",
                },
            },
            {
                "type": "blob",
                "modality": "document",
                "mime_type": "application/pdf",
                "content": "JVBERi...",
            },
        ),
        # Document URL
        (
            {
                "type": "document",
                "source": {"type": "url", "url": "https://example.com/doc.pdf"},
            },
            {"type": "uri", "modality": "document", "uri": "https://example.com/doc.pdf"},
        ),
    ],
)
def test_convert_block_multimodal(block, expected):
    assert _convert_block(block) == expected
