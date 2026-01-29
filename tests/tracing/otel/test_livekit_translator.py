import json
from unittest import mock

import pytest

from mlflow.entities.span import Span, SpanType
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.otel.translation import (
    translate_span_type_from_otel,
    translate_span_when_storing,
)
from mlflow.tracing.otel.translation.livekit import LiveKitTranslator


@pytest.mark.parametrize(
    ("attributes", "expected_type"),
    [
        ({"gen_ai.request.model": "gpt-4o-mini"}, SpanType.LLM),
        ({"gen_ai.request.model": "gpt-4"}, SpanType.LLM),
        ({"gen_ai.operation.name": "chat"}, SpanType.CHAT_MODEL),
        ({"gen_ai.operation.name": "text_completion"}, SpanType.LLM),
        ({"gen_ai.operation.name": "generate_content"}, SpanType.LLM),
        ({"gen_ai.operation.name": '"chat"'}, SpanType.CHAT_MODEL),
    ],
)
def test_livekit_span_type_from_genai_attributes(attributes, expected_type):
    translator = LiveKitTranslator()
    result = translator.translate_span_type(attributes)
    assert result == expected_type


@pytest.mark.parametrize(
    ("attributes", "expected_type"),
    [
        ({"lk.retry_count": 0}, SpanType.LLM),
        ({"lk.retry_count": 2}, SpanType.LLM),
        ({"lk.function_tool.name": "get_weather"}, SpanType.TOOL),
        ({"lk.function_tool.id": "call_123"}, SpanType.TOOL),
        ({"lk.function_tool.arguments": "{}"}, SpanType.TOOL),
        ({"lk.agent_name": "my_agent"}, SpanType.AGENT),
        ({"lk.instructions": "Be helpful"}, SpanType.AGENT),
        ({"lk.generation_id": "gen_123"}, SpanType.AGENT),
        ({"lk.tts.streaming": True}, SpanType.UNKNOWN),
        ({"lk.tts.label": "alloy"}, SpanType.UNKNOWN),
        ({"lk.input_text": "Hello world"}, SpanType.UNKNOWN),
        ({"lk.user_transcript": "Hello"}, SpanType.UNKNOWN),
        ({"lk.transcript_confidence": 0.95}, SpanType.UNKNOWN),
        ({"lk.transcription_delay": 100}, SpanType.UNKNOWN),
    ],
)
def test_livekit_span_type_inferred_from_attributes(attributes, expected_type):
    translator = LiveKitTranslator()
    result = translator.translate_span_type(attributes)
    assert result == expected_type


@pytest.mark.parametrize(
    "attributes",
    [
        {"some.other.attribute": "value"},
        {"random_key": "random_value"},
        {},
    ],
)
def test_livekit_span_type_returns_none(attributes):
    translator = LiveKitTranslator()
    result = translator.translate_span_type(attributes)
    assert result is None


@pytest.mark.parametrize(
    ("attributes", "expected_input", "expected_output"),
    [
        (
            {
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 50,
            },
            100,
            50,
        ),
        (
            {
                "gen_ai.usage.input_tokens": 200,
                "gen_ai.usage.output_tokens": 100,
            },
            200,
            100,
        ),
    ],
)
def test_livekit_token_usage_extraction(attributes, expected_input, expected_output):
    translator = LiveKitTranslator()

    assert translator.get_input_tokens(attributes) == expected_input
    assert translator.get_output_tokens(attributes) == expected_output


def test_livekit_token_usage_returns_none_when_missing():
    translator = LiveKitTranslator()
    attributes = {"some.other.attribute": "value"}

    assert translator.get_input_tokens(attributes) is None
    assert translator.get_output_tokens(attributes) is None


@pytest.mark.parametrize(
    "attributes",
    [
        {"lk.agent_name": "voice_assistant"},  
        {"lk.room_name": "my_room"},  
        {"lk.job_id": "job_123"},  
        {"lk.participant_identity": "user_456"},  
        {"lk.retry_count": 0},
        {"lk.user_input": "hello"},
        {"lk.response.text": "hi there"},
    ],
)
def test_livekit_message_format_detection(attributes):
    translator = LiveKitTranslator()
    result = translator.get_message_format(attributes)
    assert result == "livekit"


def test_livekit_message_format_returns_none_for_non_livekit():
    translator = LiveKitTranslator()
    attributes = {"some.other.attribute": "value"}
    result = translator.get_message_format(attributes)
    assert result is None


def test_livekit_get_input_from_events_with_system_and_user_messages():
    """Test EVENT_GEN_AI_SYSTEM_MESSAGE and EVENT_GEN_AI_USER_MESSAGE."""
    translator = LiveKitTranslator()
    events = [
        {
            "name": "gen_ai.system.message",
            "attributes": {"content": "You are a helpful assistant."},
        },
        {
            "name": "gen_ai.user.message",
            "attributes": {"content": "What is the capital of France?"},
        },
    ]

    result = translator.get_input_value_from_events(events)
    parsed = json.loads(result)

    assert len(parsed) == 2
    assert parsed[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert parsed[1] == {"role": "user", "content": "What is the capital of France?"}


def test_livekit_get_input_from_events_with_assistant_context():
    """Test EVENT_GEN_AI_ASSISTANT_MESSAGE for multi-turn conversations."""
    translator = LiveKitTranslator()
    events = [
        {
            "name": "gen_ai.system.message",
            "attributes": {"content": "You are a helpful assistant."},
        },
        {
            "name": "gen_ai.user.message",
            "attributes": {"content": "What is the capital of France?"},
        },
        {
            "name": "gen_ai.assistant.message",
            "attributes": {"content": "The capital of France is Paris."},
        },
        {
            "name": "gen_ai.user.message",
            "attributes": {"content": "And what about Germany?"},
        },
    ]

    result = translator.get_input_value_from_events(events)
    parsed = json.loads(result)

    assert len(parsed) == 4
    assert parsed[2] == {"role": "assistant", "content": "The capital of France is Paris."}
    assert parsed[3] == {"role": "user", "content": "And what about Germany?"}


def test_livekit_get_output_from_events_with_choice():
    """Test EVENT_GEN_AI_CHOICE for LLM responses."""
    translator = LiveKitTranslator()
    events = [
        {
            "name": "gen_ai.choice",
            "attributes": {
                "role": "assistant",
                "content": "The capital of France is Paris.",
            },
        },
    ]

    result = translator.get_output_value_from_events(events)
    parsed = json.loads(result)

    assert len(parsed) == 1
    assert parsed[0] == {
        "role": "assistant",
        "content": "The capital of France is Paris.",
    }


def test_livekit_get_output_from_events_defaults_to_assistant_role():
    translator = LiveKitTranslator()
    events = [
        {
            "name": "gen_ai.choice",
            "attributes": {"content": "Hello there!"},
        },
    ]

    result = translator.get_output_value_from_events(events)
    parsed = json.loads(result)

    assert parsed[0]["role"] == "assistant"


def test_livekit_empty_events_returns_none():
    translator = LiveKitTranslator()

    assert translator.get_input_value_from_events([]) is None
    assert translator.get_output_value_from_events([]) is None


def test_livekit_events_with_json_encoded_content():
    translator = LiveKitTranslator()
    events = [
        {
            "name": "gen_ai.user.message",
            "attributes": {"content": '"What is 2+2?"'},
        },
    ]

    result = translator.get_input_value_from_events(events)
    parsed = json.loads(result)

    # JSON-encoded content should be decoded
    assert parsed[0]["content"] == "What is 2+2?"


def test_livekit_translate_span_with_genai_events():
    """Integration test for full span translation with events."""
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.usage.input_tokens": 50,
            "gen_ai.usage.output_tokens": 100,
            "lk.agent_name": "voice_assistant",
        },
        "events": [
            {
                "name": "gen_ai.system.message",
                "attributes": {"content": "You are helpful."},
            },
            {
                "name": "gen_ai.user.message",
                "attributes": {"content": "Hello!"},
            },
            {
                "name": "gen_ai.choice",
                "attributes": {"content": "Hi there!"},
            },
        ],
    }
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    # Check inputs were extracted from events
    inputs = json.loads(result["attributes"][SpanAttributeKey.INPUTS])
    assert len(inputs) == 2
    assert inputs[0]["role"] == "system"
    assert inputs[1]["role"] == "user"

    # Check outputs were extracted from events
    outputs = json.loads(result["attributes"][SpanAttributeKey.OUTPUTS])
    assert len(outputs) == 1
    assert outputs[0]["content"] == "Hi there!"


@pytest.mark.parametrize(
    ("attributes", "expected_type"),
    [
        ({"gen_ai.operation.name": "chat"}, SpanType.CHAT_MODEL),
        ({"lk.retry_count": 0}, SpanType.LLM),
        ({"lk.function_tool.name": "weather"}, SpanType.TOOL),
    ],
)
def test_livekit_span_type_from_otel(attributes, expected_type):
    result = translate_span_type_from_otel(attributes)
    assert result == expected_type


def test_livekit_translator_detection_keys():
    """Verify detection keys match actual LiveKit attributes."""
    translator = LiveKitTranslator()

    assert "lk.agent_name" in translator.DETECTION_KEYS  # ATTR_AGENT_NAME
    assert "lk.room_name" in translator.DETECTION_KEYS  # ATTR_ROOM_NAME
    assert "lk.job_id" in translator.DETECTION_KEYS  # ATTR_JOB_ID
    assert "lk.participant_identity" in translator.DETECTION_KEYS  # ATTR_PARTICIPANT_IDENTITY


def test_livekit_translator_input_output_keys():
    """Verify input/output keys match actual LiveKit attributes."""
    translator = LiveKitTranslator()

    assert "lk.user_input" in translator.INPUT_VALUE_KEYS  # ATTR_USER_INPUT
    assert "lk.user_transcript" in translator.INPUT_VALUE_KEYS  # ATTR_USER_TRANSCRIPT
    assert "lk.chat_ctx" in translator.INPUT_VALUE_KEYS  # ATTR_CHAT_CTX
    assert "lk.input_text" in translator.INPUT_VALUE_KEYS  # ATTR_TTS_INPUT_TEXT

    assert "lk.response.text" in translator.OUTPUT_VALUE_KEYS  # ATTR_RESPONSE_TEXT
    assert "lk.response.function_calls" in translator.OUTPUT_VALUE_KEYS  # ATTR_RESPONSE_FUNCTION_CALLS


def test_livekit_translator_message_format():
    translator = LiveKitTranslator()
    assert translator.MESSAGE_FORMAT == "livekit"
