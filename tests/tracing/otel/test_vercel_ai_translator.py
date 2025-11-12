import json
from unittest import mock

import pytest

from mlflow.entities.span import Span
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.otel.translation import translate_span_when_storing


@pytest.mark.parametrize(
    ("attributes", "expected_inputs", "expected_outputs"),
    [
        # 1. generateText
        (
            {
                "ai.operationId": "ai.generateText",
                "ai.prompt": '{"prompt":"Why is the sky blue?"}',
                "ai.response.text": "Because of the scattering of light by the atmosphere.",
                "ai.response.finishReason": "length",
            },
            {
                "prompt": "Why is the sky blue?",
            },
            "Because of the scattering of light by the atmosphere.",
        ),
        # 2. generateText.doGenerate
        (
            {
                "ai.operationId": "ai.generateText.doGenerate",
                "ai.prompt.messages": (
                    '[{"role":"user","content":[{"type":"text","text":"Why is the sky blue?"}]}]'
                ),
                "ai.response.text": "Because of the scattering of light by the atmosphere.",
                "ai.response.finishReason": "length",
                "ai.response.id": "resp_0c4162a99c227acc00691324c9eaac81a3a3191fef81ca2987",
                "ai.response.model": "gpt-4-turbo-2024-04-09",
            },
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "Why is the sky blue?"}]}
                ]
            },
            {
                "text": "Because of the scattering of light by the atmosphere.",
                "finishReason": "length",
                "id": "resp_0c4162a99c227acc00691324c9eaac81a3a3191fef81ca2987",
                "model": "gpt-4-turbo-2024-04-09",
            },
        ),
        # 3. generateText with tool calls
        (
            {
                "ai.operationId": "ai.generateText.doGenerate",
                "ai.prompt.messages": (
                    '[{"role":"user","content":[{"type":"text","text":'
                    '"What is the weather in SF?"}]}]'
                ),
                "ai.prompt.tools": [
                    (
                        '{"type":"function","name":"weather","description":"Get the weather in '
                        'a location","inputSchema":{"type":"object","properties":{"location":'
                        '{"type":"string","description":"The location to get the weather for"}},'
                        '"required":["location"],"additionalProperties":false,"$schema":'
                        '"http://json-schema.org/draft-07/schema#"}}'
                    )
                ],
                "ai.prompt.toolChoice": '{"type":"auto"}',
                "ai.response.toolCalls": (
                    '[{"toolCallId":"call_PHKlxvzLK8w4PHH8CuvHXUzE","toolName":"weather",'
                    '"input":"{\\"location\\":\\"San Francisco\\"}"}]'
                ),
                "ai.response.finishReason": "tool-calls",
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "What is the weather in SF?"}],
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "weather",
                        "description": "Get the weather in a location",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location to get the weather for",
                                }
                            },
                            "required": ["location"],
                            "additionalProperties": False,
                            "$schema": "http://json-schema.org/draft-07/schema#",
                        },
                    }
                ],
                "toolChoice": {"type": "auto"},
            },
            {
                "toolCalls": [
                    {
                        "input": '{"location":"San Francisco"}',
                        "toolName": "weather",
                        "toolCallId": "call_PHKlxvzLK8w4PHH8CuvHXUzE",
                    }
                ],
                "finishReason": "tool-calls",
            },
        ),
        # 4. generateText with tool call results
        (
            {
                "ai.operationId": "ai.generateText.doGenerate",
                "ai.prompt.messages": (
                    '[{"role":"user","content":[{"type":"text",'
                    '"text":"What is the weather in San Francisco?"}]},'
                    '{"role":"assistant","content":[{"type":"tool-call","toolCallId":"call_123",'
                    '"toolName":"weather","input":{"location":"San Francisco"}}]},'
                    '{"role":"tool","content":[{"type":"tool-result","toolCallId":"call_123",'
                    '"toolName":"weather","output":{"type":"json",'
                    '"value":{"location":"San Francisco","temperature":76}}}]}]'
                ),
                "ai.prompt.toolChoice": '{"type":"auto"}',
                "ai.response.text": "The current temperature in San Francisco is 76°F.",
                "ai.response.finishReason": "stop",
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is the weather in San Francisco?"}
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool-call",
                                "toolCallId": "call_123",
                                "toolName": "weather",
                                "input": {"location": "San Francisco"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool-result",
                                "toolCallId": "call_123",
                                "toolName": "weather",
                                "output": {
                                    "type": "json",
                                    "value": {"location": "San Francisco", "temperature": 76},
                                },
                            }
                        ],
                    },
                ],
                "toolChoice": {"type": "auto"},
            },
            {
                "text": "The current temperature in San Francisco is 76°F.",
                "finishReason": "stop",
            },
        ),
        # 5. Tool execution span
        (
            {
                "ai.operationId": "ai.toolCall",
                "ai.toolCall.args": '{"location":"San Francisco"}',
                "ai.toolCall.result": '{"location":"San Francisco","temperature":76}',
            },
            {
                "location": "San Francisco",
            },
            {
                "location": "San Francisco",
                "temperature": 76,
            },
        ),
    ],
)
def test_parse_vercel_ai_generate_text(attributes, expected_inputs, expected_outputs):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {"attributes": {k: json.dumps(v) for k, v in attributes.items()}}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)
    inputs = json.loads(result["attributes"][SpanAttributeKey.INPUTS])
    assert inputs == expected_inputs
    outputs = json.loads(result["attributes"][SpanAttributeKey.OUTPUTS])
    assert outputs == expected_outputs
