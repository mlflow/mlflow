"""
This file contains unit tests for the new Gemini Python SDK
https://github.com/googleapis/python-genai
"""

import base64
import importlib.metadata
from unittest.mock import patch

import pytest
from google import genai
from packaging.version import Version

import mlflow
from mlflow.entities.span import SpanType

from tests.tracing.helper import get_traces

is_gemini_1_7_or_newer = Version(importlib.metadata.version("google.genai")) >= Version("1.7.0")

_CONTENT = {"parts": [{"text": "test answer"}], "role": "model"}

_USER_METADATA = {
    "prompt_token_count": 6,
    "candidates_token_count": 6,
    "total_token_count": 6,
    "cached_content_token_count": 0,
}


def _get_candidate(content):
    candidate = {
        "content": content,
        "avg_logprobs": 0.0,
        "finish_reason": "STOP",
        "safety_ratings": [],
        "token_count": 0,
    }

    return genai.types.Candidate(**candidate)


def _generate_content_response(content):
    res = {
        "candidates": [_get_candidate(content)],
        "usage_metadata": _USER_METADATA,
        "automatic_function_calling_history": [],
    }

    return genai.types.GenerateContentResponse(**res)


_DUMMY_GENERATE_CONTENT_RESPONSE = _generate_content_response(_CONTENT)

_DUMMY_COUNT_TOKENS_RESPONSE = {"total_count": 10}

_DUMMY_EMBEDDING_RESPONSE = {"embedding": [1, 2, 3]}

_CHAT_MESSAGES = [
    {"role": "user", "content": "test content"},
    {"role": "assistant", "content": [{"type": "text", "text": "test answer"}]},
]


def _generate_content(self, model, contents, config):
    return _DUMMY_GENERATE_CONTENT_RESPONSE


def send_message(self, content):
    return _DUMMY_GENERATE_CONTENT_RESPONSE


def count_tokens(self, model, contents):
    return _DUMMY_COUNT_TOKENS_RESPONSE


def embed_content(self, model, content):
    return _DUMMY_EMBEDDING_RESPONSE


def multiply(a: float, b: float):
    """returns a * b."""
    return a * b


TOOL_ATTRIBUTE = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "returns a * b.",
            "parameters": {
                "properties": {
                    "a": {"type": "number", "description": None, "enum": None},
                    "b": {"type": "number", "description": None, "enum": None},
                },
                "required": ["a", "b"] if is_gemini_1_7_or_newer else None,
            },
        },
    },
]


@pytest.fixture(autouse=True)
def cleanup():
    yield
    mlflow.gemini.autolog(disable=True)


def test_generate_content_enable_disable_autolog():
    with patch("google.genai.models.Models._generate_content", new=_generate_content):
        mlflow.gemini.autolog()
        client = genai.Client(api_key="dummy")
        client.models.generate_content(model="gemini-1.5-flash", contents="test content")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 2
        span = traces[0].data.spans[0]
        assert span.name == "Models.generate_content"
        assert span.span_type == SpanType.LLM
        assert span.inputs == {"contents": "test content", "model": "gemini-1.5-flash"}
        assert span.outputs == _DUMMY_GENERATE_CONTENT_RESPONSE.dict()
        assert span.get_attribute("mlflow.chat.messages") == _CHAT_MESSAGES

        span1 = traces[0].data.spans[1]
        assert span1.name == "Models._generate_content"
        assert span1.span_type == SpanType.LLM
        assert span1.inputs == {
            "contents": "test content",
            "model": "gemini-1.5-flash",
            "config": None,
        }
        assert span1.outputs == _DUMMY_GENERATE_CONTENT_RESPONSE.dict()
        assert span1.get_attribute("mlflow.chat.messages") == _CHAT_MESSAGES

        mlflow.gemini.autolog(disable=True)
        client = genai.Client(api_key="dummy")
        client.models.generate_content(model="gemini-1.5-flash", contents="test content")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_generate_content_tracing_with_error():
    with patch(
        "google.genai.models.Models._generate_content", side_effect=Exception("dummy error")
    ):
        mlflow.gemini.autolog()
        client = genai.Client(api_key="dummy")

        with pytest.raises(Exception, match="dummy error"):
            client.models.generate_content(model="gemini-1.5-flash", contents="test content")

    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans) == 2

    assert traces[0].info.status == "ERROR"
    assert traces[0].data.spans[0].status.status_code == "ERROR"
    assert traces[0].data.spans[0].status.description == "Exception: dummy error"
    assert traces[0].data.spans[1].status.status_code == "ERROR"
    assert traces[0].data.spans[1].status.description == "Exception: dummy error"


def test_generate_content_image_autolog():
    image = base64.b64encode(b"image").decode("utf-8")
    request = [
        genai.types.Part.from_bytes(mime_type="image/jpeg", data=image),
        "Caption this image",
    ]
    with patch("google.genai.models.Models._generate_content", new=_generate_content):
        mlflow.gemini.autolog()
        client = genai.Client(api_key="dummy")
        client.models.generate_content(model="gemini-1.5-flash", contents=request)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 2

    span = traces[0].data.spans[0]
    assert span.name == "Models.generate_content"
    assert span.span_type == SpanType.LLM
    assert span.inputs["model"] == "gemini-1.5-flash"
    assert span.inputs["contents"][0]["inline_data"] == {
        "data": "b'image'",
        "mime_type": "image/jpeg",
    }
    assert span.inputs["contents"][1] == "Caption this image"
    assert span.outputs == _DUMMY_GENERATE_CONTENT_RESPONSE.dict()
    assert span.get_attribute("mlflow.chat.messages") == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"detail": "auto", "url": "data:image/jpeg;base64,b'image'"},
                },
                {"type": "text", "text": "Caption this image"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "test answer"}]},
    ]

    span1 = traces[0].data.spans[1]
    assert span1.name == "Models._generate_content"
    assert span1.span_type == SpanType.LLM
    assert span1.parent_id == span.span_id
    assert span1.inputs["model"] == "gemini-1.5-flash"
    assert span1.inputs["contents"][0]["inline_data"] == {
        "data": "b'image'",
        "mime_type": "image/jpeg",
    }
    assert span1.inputs["contents"][1] == "Caption this image"
    assert span1.outputs == _DUMMY_GENERATE_CONTENT_RESPONSE.dict()


def test_generate_content_tool_calling_autolog():
    tool_call_content = {
        "parts": [
            {
                "function_call": {
                    "name": "multiply",
                    "args": {
                        "a": 57.0,
                        "b": 44.0,
                    },
                }
            }
        ],
        "role": "model",
    }

    response = _generate_content_response(tool_call_content)

    chat_messages = [
        {
            "content": "I have 57 cats, each owns 44 mittens, how many mittens is that in total?",
            "role": "user",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "multiply",
                    "type": "function",
                    "function": {"name": "multiply", "arguments": '{"a": 57.0, "b": 44.0}'},
                }
            ],
        },
    ]

    def _generate_content(self, model, contents, config):
        return response

    with patch("google.genai.models.Models._generate_content", new=_generate_content):
        mlflow.gemini.autolog()
        client = genai.Client(api_key="dummy")
        client.models.generate_content(
            model="gemini-1.5-flash",
            contents="I have 57 cats, each owns 44 mittens, how many mittens is that in total?",
            config=genai.types.GenerateContentConfig(
                tools=[multiply],
                automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 2

    span = traces[0].data.spans[0]
    assert span.name == "Models.generate_content"
    assert span.span_type == SpanType.LLM
    assert (
        span.inputs["contents"]
        == "I have 57 cats, each owns 44 mittens, how many mittens is that in total?"
    )
    assert span.get_attribute("mlflow.chat.tools") == TOOL_ATTRIBUTE
    assert span.get_attribute("mlflow.chat.messages") == chat_messages

    span1 = traces[0].data.spans[1]
    assert span1.name == "Models._generate_content"
    assert span1.span_type == SpanType.LLM
    assert span1.parent_id == span.span_id
    assert (
        span1.inputs["contents"]
        == "I have 57 cats, each owns 44 mittens, how many mittens is that in total?"
    )
    assert span1.get_attribute("mlflow.chat.tools") == TOOL_ATTRIBUTE
    assert span1.get_attribute("mlflow.chat.messages") == chat_messages


def test_generate_content_tool_calling_chat_history_autolog():
    question_content = genai.types.Content(
        **{
            "parts": [
                {
                    "text": "I have 57 cats, each owns 44 mittens, how many mittens in total?",
                }
            ],
            "role": "user",
        }
    )

    tool_call_content = genai.types.Content(
        **{
            "parts": [
                {
                    "function_call": {
                        "name": "multiply",
                        "args": {
                            "a": 57.0,
                            "b": 44.0,
                        },
                    }
                }
            ],
            "role": "model",
        }
    )

    tool_response_content = genai.types.Content(
        **{
            "parts": [{"function_response": {"name": "multiply", "response": {"result": 2508.0}}}],
            "role": "user",
        }
    )

    response = _generate_content_response(
        genai.types.Content(
            **{
                "parts": [
                    {
                        "text": "57 cats * 44 mittens/cat = 2508 mittens in total.",
                    }
                ],
                "role": "model",
            }
        )
    )

    tool_result = '{"id":null,"name":"multiply","response":{"result":2508.0}}'

    chat_messages = [
        {
            "content": [
                {
                    "type": "text",
                    "text": "I have 57 cats, each owns 44 mittens, how many mittens in total?",
                },
            ],
            "role": "user",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "multiply",
                    "type": "function",
                    "function": {"name": "multiply", "arguments": '{"a": 57.0, "b": 44.0}'},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": tool_result}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "57 cats * 44 mittens/cat = 2508 mittens in total."}
            ],
        },
    ]

    def _generate_content(self, model, contents, config):
        return response

    with patch("google.genai.models.Models._generate_content", new=_generate_content):
        mlflow.gemini.autolog()
        client = genai.Client(api_key="dummy")
        client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[question_content, tool_call_content, tool_response_content],
            config=genai.types.GenerateContentConfig(
                tools=[multiply],
                automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 2

    span = traces[0].data.spans[0]
    assert span.name == "Models.generate_content"
    assert span.span_type == SpanType.LLM
    assert span.inputs["contents"] == [
        question_content.dict(),
        tool_call_content.dict(),
        tool_response_content.dict(),
    ]
    assert span.inputs["model"] == "gemini-1.5-flash"
    assert span.get_attribute("mlflow.chat.tools") == TOOL_ATTRIBUTE
    assert span.get_attribute("mlflow.chat.messages") == chat_messages

    span1 = traces[0].data.spans[1]
    assert span1.name == "Models._generate_content"
    assert span1.span_type == SpanType.LLM
    assert span1.parent_id == span.span_id
    assert span1.inputs["contents"] == [
        question_content.dict(),
        tool_call_content.dict(),
        tool_response_content.dict(),
    ]
    assert span1.inputs["model"] == "gemini-1.5-flash"
    assert span1.get_attribute("mlflow.chat.tools") == TOOL_ATTRIBUTE
    assert span1.get_attribute("mlflow.chat.messages") == chat_messages


def test_chat_session_autolog():
    with patch("google.genai.models.Models._generate_content", new=_generate_content):
        mlflow.gemini.autolog()
        client = genai.Client(api_key="dummy")
        chat = client.chats.create(model="gemini-1.5-flash")
        chat.send_message("test content")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 3
        span = traces[0].data.spans[0]
        assert span.name == "Chat.send_message"
        assert span.span_type == SpanType.CHAT_MODEL
        assert span.inputs == {"message": "test content"}
        assert span.outputs == _DUMMY_GENERATE_CONTENT_RESPONSE.dict()
        assert span.get_attribute("mlflow.chat.messages") == _CHAT_MESSAGES

        mlflow.gemini.autolog(disable=True)
        chat = client.chats.create(model="gemini-1.5-flash")
        chat.send_message("test content")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_count_tokens_autolog():
    with patch("google.genai.models.Models.count_tokens", new=count_tokens):
        mlflow.gemini.autolog()
        client = genai.Client(api_key="dummy")
        client.models.count_tokens(model="gemini-1.5-flash", contents="test content")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1
        span = traces[0].data.spans[0]
        assert span.name == "Models.count_tokens"
        assert span.span_type == SpanType.LLM
        assert span.inputs == {"contents": "test content", "model": "gemini-1.5-flash"}
        assert span.outputs == _DUMMY_COUNT_TOKENS_RESPONSE

        mlflow.gemini.autolog(disable=True)
        client = genai.Client(api_key="dummy")
        client.models.count_tokens(model="gemini-1.5-flash", contents="test content")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_embed_content_autolog():
    with patch("google.genai.models.Models.embed_content", new=embed_content):
        mlflow.gemini.autolog()
        client = genai.Client(api_key="dummy")
        client.models.embed_content(model="text-embedding-004", content="Hello World")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1
        span = traces[0].data.spans[0]
        assert span.name == "Models.embed_content"
        assert span.span_type == SpanType.EMBEDDING
        assert span.inputs == {"content": "Hello World", "model": "text-embedding-004"}
        assert span.outputs == _DUMMY_EMBEDDING_RESPONSE

        mlflow.gemini.autolog(disable=True)
        client.models.embed_content(model="text-embedding-004", content="Hello World")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1
