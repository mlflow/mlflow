"""
This file contains unit tests for the legacy Gemini Python SDK
https://github.com/google-gemini/generative-ai-python
"""

import base64
from unittest.mock import patch

import google.generativeai as genai
import pytest
from packaging.version import Version

import mlflow
from mlflow.entities.span import SpanType

from tests.tracing.helper import get_traces

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
        "finish_reason": 0,
        "grounding_attributions": [],
        "safety_ratings": [],
        "token_count": 0,
    }

    if Version(genai.__version__) < Version("0.8.3"):
        candidate.pop("avg_logprobs")

    return candidate


def _generate_content_response(content):
    res = {
        "candidates": [_get_candidate(content)],
        "usage_metadata": _USER_METADATA,
    }

    if hasattr(genai.types.GenerateContentResponse, "model_version"):
        res["model_version"] = "gemini-1.5-flash-002"

    return res


_GENERATE_CONTENT_RESPONSE = _generate_content_response(_CONTENT)

_DUMMY_GENERATE_CONTENT_RESPONSE = genai.types.GenerateContentResponse.from_response(
    genai.protos.GenerateContentResponse(_GENERATE_CONTENT_RESPONSE)
)

_DUMMY_COUNT_TOKENS_RESPONSE = {"total_count": 10}

_DUMMY_EMBEDDING_RESPONSE = {"embedding": [1, 2, 3]}


def generate_content(self, contents):
    return _DUMMY_GENERATE_CONTENT_RESPONSE


def send_message(self, content):
    return _DUMMY_GENERATE_CONTENT_RESPONSE


def count_tokens(self, contents):
    return _DUMMY_COUNT_TOKENS_RESPONSE


def embed_content(model, content):
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
                    "a": {"type": "number", "description": "", "enum": []},
                    "b": {"type": "number", "description": "", "enum": []},
                },
                "required": ["a", "b"],
            },
        },
    },
]


@pytest.fixture(autouse=True)
def cleanup():
    yield
    mlflow.gemini.autolog(disable=True)


def test_generate_content_enable_disable_autolog():
    with patch("google.generativeai.GenerativeModel.generate_content", new=generate_content):
        mlflow.gemini.autolog()
        model = genai.GenerativeModel("gemini-1.5-flash")
        model.generate_content("test content")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1
        span = traces[0].data.spans[0]
        assert span.name == "GenerativeModel.generate_content"
        assert span.span_type == SpanType.LLM
        assert span.inputs == {"contents": "test content"}
        assert span.outputs == _GENERATE_CONTENT_RESPONSE

        mlflow.gemini.autolog(disable=True)
        model = genai.GenerativeModel("gemini-1.5-flash")
        model.generate_content("test content")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_generate_content_tracing_with_error():
    with patch(
        "google.generativeai.GenerativeModel.generate_content", side_effect=Exception("dummy error")
    ):
        mlflow.gemini.autolog()
        model = genai.GenerativeModel("gemini-1.5-flash")

        with pytest.raises(Exception, match="dummy error"):
            model.generate_content("test content")

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    assert traces[0].data.spans[0].status.status_code == "ERROR"
    assert traces[0].data.spans[0].status.description == "Exception: dummy error"


def test_generate_content_image_autolog():
    image = base64.b64encode(b"image").decode("utf-8")
    request = [{"mime_type": "image/jpeg", "data": image}, "Caption this image"]
    with patch("google.generativeai.GenerativeModel.generate_content", new=generate_content):
        mlflow.gemini.autolog()
        model = genai.GenerativeModel("gemini-1.5-flash")
        model.generate_content(request)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "GenerativeModel.generate_content"
    assert span.span_type == SpanType.LLM
    assert span.inputs == {"contents": request}
    assert span.outputs == _GENERATE_CONTENT_RESPONSE


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

    raw_response = _generate_content_response(tool_call_content)
    response = genai.types.GenerateContentResponse.from_response(
        genai.protos.GenerateContentResponse(raw_response)
    )

    def generate_content(self, content):
        return response

    with patch("google.generativeai.GenerativeModel.generate_content", new=generate_content):
        mlflow.gemini.autolog()
        model = genai.GenerativeModel("gemini-1.5-flash", tools=[multiply])
        model.generate_content(
            "I have 57 cats, each owns 44 mittens, how many mittens is that in total?"
        )

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "GenerativeModel.generate_content"
    assert span.span_type == SpanType.LLM
    assert span.inputs == {
        "content": "I have 57 cats, each owns 44 mittens, how many mittens is that in total?"
    }
    assert span.get_attribute("mlflow.chat.tools") == TOOL_ATTRIBUTE


def test_generate_content_tool_calling_chat_history_autolog():
    question_content = genai.protos.Content(
        {
            "parts": [
                {
                    "text": "I have 57 cats, each owns 44 mittens, how many mittens in total?",
                }
            ],
            "role": "user",
        }
    )

    tool_call_content = genai.protos.Content(
        {
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

    tool_response_content = genai.protos.Content(
        {
            "parts": [{"function_response": {"name": "multiply", "response": {"result": 2508.0}}}],
            "role": "user",
        }
    )

    raw_response = _generate_content_response(
        genai.protos.Content(
            {
                "parts": [
                    {
                        "text": "57 cats * 44 mittens/cat = 2508 mittens in total.",
                    }
                ],
                "role": "model",
            }
        )
    )

    response = genai.types.GenerateContentResponse.from_response(
        genai.protos.GenerateContentResponse(raw_response)
    )

    def generate_content(self, content):
        return response

    with patch("google.generativeai.GenerativeModel.generate_content", new=generate_content):
        mlflow.gemini.autolog()
        model = genai.GenerativeModel("gemini-1.5-flash", tools=[multiply])
        model.generate_content([question_content, tool_call_content, tool_response_content])

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "GenerativeModel.generate_content"
    assert span.span_type == SpanType.LLM
    assert span.inputs == {
        "content": [str(question_content), str(tool_call_content), str(tool_response_content)]
    }
    assert span.get_attribute("mlflow.chat.tools") == TOOL_ATTRIBUTE


def test_chat_session_autolog():
    with patch("google.generativeai.ChatSession.send_message", new=send_message):
        mlflow.gemini.autolog()
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(history=[])
        chat.send_message("test content")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1
        span = traces[0].data.spans[0]
        assert span.name == "ChatSession.send_message"
        assert span.span_type == SpanType.CHAT_MODEL
        assert span.inputs == {"content": "test content"}
        assert span.outputs == _GENERATE_CONTENT_RESPONSE

        mlflow.gemini.autolog(disable=True)
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(history=[])
        chat.send_message("test content")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_count_tokens_autolog():
    with patch("google.generativeai.GenerativeModel.count_tokens", new=count_tokens):
        mlflow.gemini.autolog()
        model = genai.GenerativeModel("gemini-1.5-flash")
        model.count_tokens("test content")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1
        span = traces[0].data.spans[0]
        assert span.name == "GenerativeModel.count_tokens"
        assert span.span_type == SpanType.LLM
        assert span.inputs == {"contents": "test content"}
        assert span.outputs == _DUMMY_COUNT_TOKENS_RESPONSE

        mlflow.gemini.autolog(disable=True)
        model = genai.GenerativeModel("gemini-1.5-flash")
        model.count_tokens("test content")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_embed_content_autolog():
    with patch("google.generativeai.embed_content", new=embed_content):
        mlflow.gemini.autolog()
        genai.embed_content(model="models/text-embedding-004", content="Hello World")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 1
        span = traces[0].data.spans[0]
        assert span.name == "embed_content"
        assert span.span_type == SpanType.EMBEDDING
        assert span.inputs == {"content": "Hello World", "model": "models/text-embedding-004"}
        assert span.outputs == _DUMMY_EMBEDDING_RESPONSE

        mlflow.gemini.autolog(disable=True)
        genai.embed_content(model="models/text-embedding-004", content="Hello World")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1
