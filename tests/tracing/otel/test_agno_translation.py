import json
from unittest import mock

import pytest

from mlflow.entities.span import Span, SpanType
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.otel.translation import (
    translate_span_type_from_otel,
    translate_span_when_storing,
)
from mlflow.tracing.otel.translation.agno import AgnoTranslator


@pytest.mark.parametrize(
    ("attributes", "expected_format"),
    [
        ({"agno.agent.id": "agent_123"}, "agno"),
        ({"agno.team.id": "team_456"}, "agno"),
        ({"agno.model.id": "gemini-2.0-flash-thinking"}, "agno"),
        ({"agno.run.id": "run_789"}, "agno"),
        ({"agno.agent.id": "agent_1", "openinference.span.kind": "AGENT"}, "agno"),
        ({"agno.agent.id": "a", "agno.run.id": "r", "other.attr": "val"}, "agno"),
    ],
)
def test_agno_detects_agno_prefixed_attributes(attributes, expected_format):
    translator = AgnoTranslator()
    result = translator.get_message_format(attributes)
    assert result == expected_format


@pytest.mark.parametrize(
    "attributes",
    [
        # Generic LLM messages - no agno.* prefix
        {
            "openinference.span.kind": "LLM",
            "input.value": json.dumps({"messages": [{"role": "user", "content": "Hi"}]}),
        },
        # Messages with Agno-like fields but no agno.* prefix
        {
            "openinference.span.kind": "LLM",
            "input.value": json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hi",
                            "from_history": True,
                            "stop_after_tool_call": False,
                        }
                    ]
                }
            ),
        },
        # Unknown span kind
        {"openinference.span.kind": "UNKNOWN"},
        # No relevant attributes
        {"some.other.attribute": "value"},
        # Empty attributes
        {},
    ],
)
def test_agno_returns_none_without_agno_prefix(attributes):
    translator = AgnoTranslator()
    result = translator.get_message_format(attributes)
    assert result is None


def test_agno_translator_inherits_from_open_inference():
    from mlflow.tracing.otel.translation.open_inference import OpenInferenceTranslator

    translator = AgnoTranslator()
    assert isinstance(translator, OpenInferenceTranslator)


def test_agno_translator_message_format_constant():
    translator = AgnoTranslator()
    assert translator.MESSAGE_FORMAT == "agno"


def test_agno_translator_attribute_prefix():
    translator = AgnoTranslator()
    assert translator.AGNO_ATTRIBUTE_PREFIX == "agno."


@pytest.mark.parametrize(
    ("attributes", "expected_type"),
    [
        ({"openinference.span.kind": "LLM"}, SpanType.LLM),
        ({"openinference.span.kind": "AGENT"}, SpanType.AGENT),
        ({"openinference.span.kind": "CHAIN"}, SpanType.CHAIN),
        ({"openinference.span.kind": "TOOL"}, SpanType.TOOL),
        ({"openinference.span.kind": json.dumps("LLM")}, SpanType.LLM),
        ({"openinference.span.kind": json.dumps("AGENT")}, SpanType.AGENT),
    ],
)
def test_agno_span_type_from_otel(attributes, expected_type):
    result = translate_span_type_from_otel(attributes)
    assert result == expected_type


def test_agno_translate_span_sets_message_format():
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {
        "attributes": {
            "openinference.span.kind": "LLM",
            "agno.agent.id": "agent_123",
            "input.value": json.dumps({"messages": [{"role": "user", "content": "Hi"}]}),
            "output.value": json.dumps("Hello!"),
        }
    }
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    assert SpanAttributeKey.MESSAGE_FORMAT in result["attributes"]
    message_format = json.loads(result["attributes"][SpanAttributeKey.MESSAGE_FORMAT])
    assert message_format == "agno"


def test_agno_translate_span_sets_inputs_outputs():
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    input_messages = {"messages": [{"role": "user", "content": "Hello"}]}
    span_dict = {
        "attributes": {
            "openinference.span.kind": "LLM",
            "agno.agent.id": "agent_123",
            "input.value": json.dumps(input_messages),
            "output.value": json.dumps("Hi there!"),
        }
    }
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    inputs = json.loads(result["attributes"][SpanAttributeKey.INPUTS])
    assert inputs == input_messages

    outputs = json.loads(result["attributes"][SpanAttributeKey.OUTPUTS])
    assert outputs == "Hi there!"


def test_agno_does_not_match_similar_frameworks():
    translator = AgnoTranslator()

    # LangChain-like
    assert translator.get_message_format({"langchain.run_type": "llm"}) is None

    # CrewAI-like
    assert translator.get_message_format({"crewai.agent.role": "researcher"}) is None

    # Generic OpenInference
    assert (
        translator.get_message_format(
            {
                "openinference.span.kind": "LLM",
                "llm.model_name": "gpt-4",
            }
        )
        is None
    )
