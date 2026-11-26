# Tests for the new Gemini Python SDK:
# https://github.com/googleapis/python-genai

import asyncio
import base64
import copy
import gc
import importlib.metadata
import json
import re
from unittest.mock import patch

import pytest
from google import genai
from packaging.version import Version

import mlflow
from mlflow.entities import SpanLogLevel
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.version import IS_TRACING_SDK_ONLY

from tests.tracing.helper import get_traces

google_gemini_version = Version(importlib.metadata.version("google.genai"))
is_gemini_1_7_or_newer = google_gemini_version >= Version("1.7.0")

_CONTENT = {"parts": [{"text": "test answer"}], "role": "model"}

_USER_METADATA = {
    "prompt_token_count": 6,
    "candidates_token_count": 6,
    "total_token_count": 12,
    "cached_content_token_count": 0,
}

_USER_METADATA_WITH_CACHE = {
    "prompt_token_count": 50,
    "candidates_token_count": 20,
    "total_token_count": 70,
    "cached_content_token_count": 30,
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


def _generate_content_response(content, usage_metadata=None):
    res = {
        "candidates": [_get_candidate(content)],
        "usage_metadata": usage_metadata or _USER_METADATA,
        "automatic_function_calling_history": [],
    }

    return genai.types.GenerateContentResponse(**res)


_DUMMY_GENERATE_CONTENT_RESPONSE = _generate_content_response(_CONTENT)

_STREAM_CHUNKS = [
    _generate_content_response({"parts": [{"text": "test "}], "role": "model"}),
    _generate_content_response({"parts": [{"text": "answer"}], "role": "model"}),
]


_FUNCTION_CALL_CHUNK = _generate_content_response({
    "parts": [{"function_call": {"name": "multiply", "args": {"a": 57.0, "b": 44.0}}}],
    "role": "model",
})

_FUNCTION_ANSWER_CHUNK = _generate_content_response({
    "parts": [{"text": "57 * 44 = 2508"}],
    "role": "model",
})


def _expected_streaming_output():
    response = _STREAM_CHUNKS[-1].model_copy(deep=True)
    response.candidates[0].content.parts = [genai.types.Part(text="test answer")]
    return response.model_dump()


_DUMMY_COUNT_TOKENS_RESPONSE = {"total_count": 10}

_DUMMY_EMBEDDING_RESPONSE = {"embedding": [1, 2, 3]}


def _dummy_generate_content(is_async: bool):
    if is_async:

        async def _generate_content(self, model, contents, config):
            return _DUMMY_GENERATE_CONTENT_RESPONSE

    else:

        def _generate_content(self, model, contents, config):
            return _DUMMY_GENERATE_CONTENT_RESPONSE

    return _generate_content


def _dummy_generate_content_stream(is_async: bool, error: Exception | None = None):
    if is_async:

        async def _generate_content_stream(self, model, contents, config):
            async def _stream():
                yield _STREAM_CHUNKS[0]
                yield _STREAM_CHUNKS[1]
                if error:
                    raise error

            return _stream()

    else:

        def _generate_content_stream(self, model, contents, config):
            yield _STREAM_CHUNKS[0]
            yield _STREAM_CHUNKS[1]
            if error:
                raise error

    return _generate_content_stream


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


@pytest.fixture(params=[True, False], ids=["async", "sync"])
def is_async(request):
    return request.param


def _call_generate_content(
    is_async: bool, contents: str, model: str = "gemini-1.5-flash", config=None
):
    client = genai.Client(api_key="dummy")
    if is_async:
        return asyncio.run(
            client.aio.models.generate_content(model=model, contents=contents, config=config)
        )
    else:
        return client.models.generate_content(model=model, contents=contents, config=config)


def _create_chat_and_send_message(is_async: bool, message: str):
    client = genai.Client(api_key="dummy")
    if is_async:
        chat = client.aio.chats.create(model="gemini-1.5-flash")
        return asyncio.run(chat.send_message(message))
    else:
        chat = client.chats.create(model="gemini-1.5-flash")
        return chat.send_message(message)


def _call_generate_content_stream(is_async: bool, contents: str = "test content", config=None):
    client = genai.Client(api_key="dummy")
    kwargs = {"model": "gemini-1.5-flash", "contents": contents}
    if config is not None:
        kwargs["config"] = config
    if is_async:

        async def _consume():
            stream = await client.aio.models.generate_content_stream(**kwargs)
            return [chunk async for chunk in stream]

        return asyncio.run(_consume())
    return list(client.models.generate_content_stream(**kwargs))


def _create_chat_and_send_message_stream(is_async: bool):
    client = genai.Client(api_key="dummy")
    if is_async:

        async def _consume():
            chat = client.aio.chats.create(model="gemini-1.5-flash")
            stream = await chat.send_message_stream("test content")
            return [chunk async for chunk in stream]

        return asyncio.run(_consume())
    chat = client.chats.create(model="gemini-1.5-flash")
    return list(chat.send_message_stream("test content"))


def test_generate_content_enable_disable_autolog(is_async, mock_litellm_cost):
    cls = "AsyncModels" if is_async else "Models"
    with (
        patch(
            f"google.genai.models.{cls}._generate_content", new=_dummy_generate_content(is_async)
        ),
    ):
        mlflow.gemini.autolog()
        _call_generate_content(is_async, "test content")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 2

        span = traces[0].data.spans[0]
        assert span.name == f"{cls}.generate_content"
        assert span.span_type == SpanType.LLM
        assert span.inputs == {
            "contents": "test content",
            "model": "gemini-1.5-flash",
            "config": None,
        }
        assert span.outputs == _DUMMY_GENERATE_CONTENT_RESPONSE.model_dump()
        assert span.model_name == "gemini-1.5-flash"

        span1 = traces[0].data.spans[1]
        assert span1.name == f"{cls}._generate_content"
        assert span1.span_type == SpanType.LLM
        assert span1.inputs == {
            "contents": "test content",
            "model": "gemini-1.5-flash",
            "config": None,
        }
        assert span1.outputs == _DUMMY_GENERATE_CONTENT_RESPONSE.model_dump()

        assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
            TokenUsageKey.INPUT_TOKENS: 6,
            TokenUsageKey.OUTPUT_TOKENS: 6,
            TokenUsageKey.TOTAL_TOKENS: 12,
            TokenUsageKey.CACHE_READ_INPUT_TOKENS: 0,
        }

        if not IS_TRACING_SDK_ONLY:
            # Verify cost is calculated (6 input tokens * 1.0 + 6 output tokens * 2.0)
            assert span.llm_cost == {
                "input_cost": 6.0,
                "output_cost": 12.0,
                "total_cost": 18.0,
            }

        assert traces[0].info.token_usage == {
            "input_tokens": 6,
            "output_tokens": 6,
            "total_tokens": 12,
            "cache_read_input_tokens": 0,
        }

        mlflow.gemini.autolog(disable=True)
        _call_generate_content(is_async, "test content")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_generate_content_tracing_with_error(is_async):
    if is_async:

        async def _generate_content(self, model, contents, config):
            raise Exception("dummy error")

    else:

        def _generate_content(self, model, contents, config):
            raise Exception("dummy error")

    cls = "AsyncModels" if is_async else "Models"
    with patch(f"google.genai.models.{cls}._generate_content", new=_generate_content):
        mlflow.gemini.autolog()

        with pytest.raises(Exception, match="dummy error"):
            _call_generate_content(is_async, "test content")

    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans) == 2

    assert traces[0].info.status == "ERROR"
    assert traces[0].data.spans[0].status.status_code == "ERROR"
    assert traces[0].data.spans[0].status.description == "Exception: dummy error"
    assert traces[0].data.spans[1].status.status_code == "ERROR"
    assert traces[0].data.spans[1].status.description == "Exception: dummy error"


def test_generate_content_image_autolog(mock_litellm_cost):
    image = base64.b64encode(b"image").decode("utf-8")
    request = [
        genai.types.Part.from_bytes(mime_type="image/jpeg", data=image),
        "Caption this image",
    ]
    cls = "AsyncModels" if is_async else "Models"
    with patch(
        f"google.genai.models.{cls}._generate_content", new=_dummy_generate_content(is_async)
    ):
        mlflow.gemini.autolog()
        _call_generate_content(is_async, request)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 2

    span = traces[0].data.spans[0]
    assert span.name == f"{cls}.generate_content"
    assert span.span_type == SpanType.LLM
    assert span.log_level == SpanLogLevel.INFO
    assert span.inputs["model"] == "gemini-1.5-flash"
    extra = {"display_name": None} if google_gemini_version >= Version("1.15.0") else {}
    inline_data = span.inputs["contents"][0]["inline_data"]
    assert inline_data["mime_type"] == "image/jpeg"
    # Auto-extraction replaces bytes repr with mlflow-attachment:// URI
    assert inline_data["data"].startswith("mlflow-attachment://")
    assert "content_type=image%2Fjpeg" in inline_data["data"]
    if extra:
        assert inline_data["display_name"] is None
    assert span.inputs["contents"][1] == "Caption this image"
    assert span.outputs == _DUMMY_GENERATE_CONTENT_RESPONSE.model_dump()
    assert span.model_name == "gemini-1.5-flash"

    span1 = traces[0].data.spans[1]
    assert span1.name == f"{cls}._generate_content"
    assert span1.span_type == SpanType.LLM
    assert span1.parent_id == span.span_id
    assert span1.inputs["model"] == "gemini-1.5-flash"
    inline_data1 = span1.inputs["contents"][0]["inline_data"]
    assert inline_data1["mime_type"] == "image/jpeg"
    assert inline_data1["data"].startswith("mlflow-attachment://")
    assert "content_type=image%2Fjpeg" in inline_data1["data"]
    if extra:
        assert inline_data1["display_name"] is None
    assert span1.inputs["contents"][1] == "Caption this image"
    assert span1.outputs == _DUMMY_GENERATE_CONTENT_RESPONSE.model_dump()

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 6,
        TokenUsageKey.OUTPUT_TOKENS: 6,
        TokenUsageKey.TOTAL_TOKENS: 12,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 0,
    }
    if not IS_TRACING_SDK_ONLY:
        assert span.llm_cost == {
            "input_cost": 6.0,
            "output_cost": 12.0,
            "total_cost": 18.0,
        }

    assert traces[0].info.token_usage == {
        "input_tokens": 6,
        "output_tokens": 6,
        "total_tokens": 12,
        "cache_read_input_tokens": 0,
    }


def test_generate_content_tool_calling_autolog(is_async, mock_litellm_cost):
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
    if is_async:

        async def _generate_content(self, model, contents, config):
            return response

    else:

        def _generate_content(self, model, contents, config):
            return response

    cls = "AsyncModels" if is_async else "Models"

    with patch(f"google.genai.models.{cls}._generate_content", new=_generate_content):
        mlflow.gemini.autolog()
        _call_generate_content(
            is_async,
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
    assert span.name == f"{cls}.generate_content"
    assert span.span_type == SpanType.LLM
    assert (
        span.inputs["contents"]
        == "I have 57 cats, each owns 44 mittens, how many mittens is that in total?"
    )
    assert span.get_attribute(SpanAttributeKey.CHAT_TOOLS) == TOOL_ATTRIBUTE
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "gemini"
    assert span.model_name == "gemini-1.5-flash"

    span1 = traces[0].data.spans[1]
    assert span1.name == f"{cls}._generate_content"
    assert span1.span_type == SpanType.LLM
    assert span1.parent_id == span.span_id
    assert (
        span1.inputs["contents"]
        == "I have 57 cats, each owns 44 mittens, how many mittens is that in total?"
    )
    assert span1.get_attribute(SpanAttributeKey.CHAT_TOOLS) == TOOL_ATTRIBUTE
    assert span1.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "gemini"

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 6,
        TokenUsageKey.OUTPUT_TOKENS: 6,
        TokenUsageKey.TOTAL_TOKENS: 12,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 0,
    }
    if not IS_TRACING_SDK_ONLY:
        assert span.llm_cost == {
            "input_cost": 6.0,
            "output_cost": 12.0,
            "total_cost": 18.0,
        }

    assert traces[0].info.token_usage == {
        "input_tokens": 6,
        "output_tokens": 6,
        "total_tokens": 12,
        "cache_read_input_tokens": 0,
    }


def test_generate_content_tool_calling_chat_history_autolog(is_async, mock_litellm_cost):
    question_content = genai.types.Content(**{
        "parts": [
            {
                "text": "I have 57 cats, each owns 44 mittens, how many mittens in total?",
            }
        ],
        "role": "user",
    })

    tool_call_content = genai.types.Content(**{
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
    })

    tool_response_content = genai.types.Content(**{
        "parts": [{"function_response": {"name": "multiply", "response": {"result": 2508.0}}}],
        "role": "user",
    })

    response = _generate_content_response(
        genai.types.Content(**{
            "parts": [
                {
                    "text": "57 cats * 44 mittens/cat = 2508 mittens in total.",
                }
            ],
            "role": "model",
        })
    )

    cls = "AsyncModels" if is_async else "Models"

    if is_async:

        async def _generate_content(self, model, contents, config):
            return response

    else:

        def _generate_content(self, model, contents, config):
            return response

    with patch(f"google.genai.models.{cls}._generate_content", new=_generate_content):
        mlflow.gemini.autolog()
        _call_generate_content(
            is_async,
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
    assert span.name == f"{cls}.generate_content"
    assert span.span_type == SpanType.LLM
    assert span.inputs["contents"] == [
        question_content.model_dump(),
        tool_call_content.model_dump(),
        tool_response_content.model_dump(),
    ]
    assert span.inputs["model"] == "gemini-1.5-flash"
    assert span.get_attribute("mlflow.chat.tools") == TOOL_ATTRIBUTE
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "gemini"
    assert span.model_name == "gemini-1.5-flash"

    span1 = traces[0].data.spans[1]
    assert span1.name == f"{cls}._generate_content"
    assert span1.span_type == SpanType.LLM
    assert span1.parent_id == span.span_id
    assert span1.inputs["contents"] == [
        question_content.model_dump(),
        tool_call_content.model_dump(),
        tool_response_content.model_dump(),
    ]
    assert span1.inputs["model"] == "gemini-1.5-flash"
    assert span1.get_attribute("mlflow.chat.tools") == TOOL_ATTRIBUTE
    assert span1.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "gemini"

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 6,
        TokenUsageKey.OUTPUT_TOKENS: 6,
        TokenUsageKey.TOTAL_TOKENS: 12,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 0,
    }
    if not IS_TRACING_SDK_ONLY:
        assert span.llm_cost == {
            "input_cost": 6.0,
            "output_cost": 12.0,
            "total_cost": 18.0,
        }

    assert traces[0].info.token_usage == {
        "input_tokens": 6,
        "output_tokens": 6,
        "total_tokens": 12,
        "cache_read_input_tokens": 0,
    }


def test_chat_session_autolog(is_async):
    cls = "AsyncModels" if is_async else "Models"
    with patch(
        f"google.genai.models.{cls}._generate_content", new=_dummy_generate_content(is_async)
    ):
        mlflow.gemini.autolog()
        _create_chat_and_send_message(is_async, "test content")

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 3
        span = traces[0].data.spans[0]
        assert span.name == "AsyncChat.send_message" if is_async else "Chat.send_message"
        assert span.span_type == SpanType.CHAT_MODEL
        assert span.inputs == {"message": "test content"}
        assert span.outputs == _DUMMY_GENERATE_CONTENT_RESPONSE.model_dump()
        assert span.model_name == "gemini-1.5-flash"

        mlflow.gemini.autolog(disable=True)
        _create_chat_and_send_message(is_async, "test content")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_generate_content_stream_autolog(is_async):
    cls = "AsyncModels" if is_async else "Models"
    with patch(
        f"google.genai.models.{cls}._generate_content_stream",
        new=_dummy_generate_content_stream(is_async),
    ):
        mlflow.gemini.autolog()
        chunks = _call_generate_content_stream(is_async)

        traces = get_traces()
        assert len(traces) == 1
        assert traces[0].info.status == "OK"
        assert len(traces[0].data.spans) == 2

        span = traces[0].data.spans[0]
        assert span.name == f"{cls}.generate_content_stream"
        assert span.span_type == SpanType.LLM
        assert span.inputs == {
            "contents": "test content",
            "model": "gemini-1.5-flash",
        }
        assert span.outputs == _expected_streaming_output()
        assert span.model_name == "gemini-1.5-flash"
        assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
            TokenUsageKey.INPUT_TOKENS: 6,
            TokenUsageKey.OUTPUT_TOKENS: 6,
            TokenUsageKey.TOTAL_TOKENS: 12,
            TokenUsageKey.CACHE_READ_INPUT_TOKENS: 0,
        }
        assert [event.name for event in span.events] == [
            "mlflow.chunk.item.0",
            "mlflow.chunk.item.1",
        ]
        for event, chunk in zip(span.events, chunks):
            assert json.loads(event.attributes["mlflow.chunk.value"]) == chunk.model_dump()

        child = traces[0].data.spans[1]
        assert child.name == f"{cls}._generate_content_stream"
        assert child.parent_id == span.span_id
        assert child.outputs == _expected_streaming_output()

        mlflow.gemini.autolog(disable=True)
        _call_generate_content_stream(is_async)

        traces = get_traces()
        assert len(traces) == 1


def test_chat_send_message_stream_autolog(is_async):
    cls = "AsyncModels" if is_async else "Models"
    chat_cls = "AsyncChat" if is_async else "Chat"
    with patch(
        f"google.genai.models.{cls}._generate_content_stream",
        new=_dummy_generate_content_stream(is_async),
    ):
        mlflow.gemini.autolog()
        chunks = _create_chat_and_send_message_stream(is_async)

    assert [chunk.text for chunk in chunks] == ["test ", "answer"]
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 3

    chat_span, model_span, request_span = traces[0].data.spans
    assert chat_span.name == f"{chat_cls}.send_message_stream"
    assert chat_span.span_type == SpanType.CHAT_MODEL
    assert chat_span.inputs == {"message": "test content"}
    assert chat_span.outputs == _expected_streaming_output()
    assert chat_span.model_name == "gemini-1.5-flash"

    assert model_span.name == f"{cls}.generate_content_stream"
    assert model_span.parent_id == chat_span.span_id
    assert request_span.name == f"{cls}._generate_content_stream"
    assert request_span.parent_id == model_span.span_id


def test_generate_content_stream_with_error(is_async):
    cls = "AsyncModels" if is_async else "Models"
    with patch(
        f"google.genai.models.{cls}._generate_content_stream",
        new=_dummy_generate_content_stream(is_async, error=Exception("stream error")),
    ):
        mlflow.gemini.autolog()
        with pytest.raises(Exception, match="stream error"):
            _call_generate_content_stream(is_async)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    assert len(traces[0].data.spans) == 2
    for span in traces[0].data.spans:
        assert span.status.status_code == "ERROR"
        assert span.status.description == "Exception: stream error"
        assert span.outputs == _expected_streaming_output()


def test_generate_content_stream_automatic_function_calling(is_async):
    cls = "AsyncModels" if is_async else "Models"
    requests = []

    def _next_response(contents):
        requests.append(contents)
        return _FUNCTION_CALL_CHUNK if len(requests) == 1 else _FUNCTION_ANSWER_CHUNK

    if is_async:

        async def _generate_content_stream(self, model, contents, config):
            chunk = _next_response(contents)

            async def _stream():
                yield chunk

            return _stream()

    else:

        def _generate_content_stream(self, model, contents, config):
            yield _next_response(contents)

    config = genai.types.GenerateContentConfig(tools=[multiply])
    client = genai.Client(api_key="dummy")
    with patch(f"google.genai.models.{cls}._generate_content_stream", new=_generate_content_stream):
        mlflow.gemini.autolog()
        if is_async:

            async def _consume():
                stream = await client.aio.models.generate_content_stream(
                    model="gemini-1.5-flash", contents="What is 57 * 44?", config=config
                )
                return [chunk async for chunk in stream]

            chunks = asyncio.run(_consume())
        else:
            chunks = list(
                client.models.generate_content_stream(
                    model="gemini-1.5-flash", contents="What is 57 * 44?", config=config
                )
            )

    assert len(requests) == 2
    # Newer SDK versions also yield the function call chunk to the caller, older ones don't.
    assert chunks[-1].text == "57 * 44 = 2508"
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"

    stream_span, function_call_span, answer_span = traces[0].data.spans
    assert stream_span.name == f"{cls}.generate_content_stream"
    assert [event.name for event in stream_span.events] == [
        f"mlflow.chunk.item.{i}" for i in range(len(chunks))
    ]
    assert (
        stream_span.outputs["candidates"][0]["content"]["parts"][-1]
        == _FUNCTION_ANSWER_CHUNK.candidates[0].content.parts[0].model_dump()
    )
    assert stream_span.get_attribute(SpanAttributeKey.CHAT_TOOLS) == TOOL_ATTRIBUTE

    assert function_call_span.name == f"{cls}._generate_content_stream"
    assert function_call_span.parent_id == stream_span.span_id
    assert function_call_span.outputs == _FUNCTION_CALL_CHUNK.model_dump()

    assert answer_span.name == f"{cls}._generate_content_stream"
    assert answer_span.parent_id == stream_span.span_id
    assert answer_span.outputs == _FUNCTION_ANSWER_CHUNK.model_dump()


def test_generate_content_stream_chunk_recording_failure(is_async):
    cls = "AsyncModels" if is_async else "Models"
    with (
        patch(
            f"google.genai.models.{cls}._generate_content_stream",
            new=_dummy_generate_content_stream(is_async),
        ),
        patch("mlflow.gemini.stream.SpanEvent", side_effect=ValueError("bad chunk")) as mock_event,
    ):
        mlflow.gemini.autolog()
        chunks = _call_generate_content_stream(is_async)

    assert mock_event.call_count == 4
    assert [chunk.text for chunk in chunks] == ["test ", "answer"]
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    for span in traces[0].data.spans:
        assert span.events == []
        assert span.outputs == _expected_streaming_output()


def test_generate_content_stream_close(is_async):
    cls = "AsyncModels" if is_async else "Models"
    with patch(
        f"google.genai.models.{cls}._generate_content_stream",
        new=_dummy_generate_content_stream(is_async),
    ):
        mlflow.gemini.autolog()
        client = genai.Client(api_key="dummy")
        if is_async:

            async def _close_stream():
                stream = await client.aio.models.generate_content_stream(
                    model="gemini-1.5-flash", contents="test content"
                )
                assert (await stream.__anext__()).text == "test "
                await stream.aclose()

            asyncio.run(_close_stream())
        else:
            stream = client.models.generate_content_stream(
                model="gemini-1.5-flash", contents="test content"
            )
            assert next(stream).text == "test "
            stream.close()

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 2
    for span in traces[0].data.spans:
        assert span.outputs["candidates"][0]["content"]["parts"] == [
            _STREAM_CHUNKS[0].candidates[0].content.parts[0].model_dump()
        ]


def test_generate_content_stream_abandoned(is_async):
    # Stopping early without close() is common (e.g. a user cancels a chat response). The span
    # must still end once the stream is released, otherwise the trace is never exported.
    cls = "AsyncModels" if is_async else "Models"
    with patch(
        f"google.genai.models.{cls}._generate_content_stream",
        new=_dummy_generate_content_stream(is_async),
    ):
        mlflow.gemini.autolog()
        client = genai.Client(api_key="dummy")
        if is_async:

            async def _abandon_stream():
                stream = await client.aio.models.generate_content_stream(
                    model="gemini-1.5-flash", contents="test content"
                )
                async for _ in stream:
                    break
                assert get_traces() == []
                del stream

            asyncio.run(_abandon_stream())
        else:
            stream = client.models.generate_content_stream(
                model="gemini-1.5-flash", contents="test content"
            )
            for _ in stream:
                break
            assert get_traces() == []
            del stream

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 2
    for span in traces[0].data.spans:
        assert span.end_time_ns is not None
        assert span.outputs["candidates"][0]["content"]["parts"] == [
            _STREAM_CHUNKS[0].candidates[0].content.parts[0].model_dump()
        ]


def test_generate_content_stream_never_consumed(is_async):
    cls = "AsyncModels" if is_async else "Models"
    with patch(
        f"google.genai.models.{cls}._generate_content_stream",
        new=_dummy_generate_content_stream(is_async),
    ):
        mlflow.gemini.autolog()
        client = genai.Client(api_key="dummy")
        if is_async:

            async def _drop_stream():
                stream = await client.aio.models.generate_content_stream(
                    model="gemini-1.5-flash", contents="test content"
                )
                del stream

            asyncio.run(_drop_stream())
        else:
            stream = client.models.generate_content_stream(
                model="gemini-1.5-flash", contents="test content"
            )
            del stream

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    (span,) = traces[0].data.spans
    assert span.name == f"{cls}.generate_content_stream"
    assert span.outputs is None


def test_generate_content_stream_abandoned_releases_stream(is_async):
    # Ending the span of an abandoned stream must not keep the SDK's stream, which holds the
    # HTTP response, alive.
    cls = "AsyncModels" if is_async else "Models"
    released = []

    if is_async:

        async def _generate_content_stream(self, model, contents, config):
            async def _stream():
                try:
                    yield _STREAM_CHUNKS[0]
                    yield _STREAM_CHUNKS[1]
                finally:
                    released.append(True)

            return _stream()

    else:

        def _generate_content_stream(self, model, contents, config):
            try:
                yield _STREAM_CHUNKS[0]
                yield _STREAM_CHUNKS[1]
            finally:
                released.append(True)

    with patch(f"google.genai.models.{cls}._generate_content_stream", new=_generate_content_stream):
        mlflow.gemini.autolog()
        client = genai.Client(api_key="dummy")
        if is_async:

            async def _abandon_stream():
                stream = await client.aio.models.generate_content_stream(
                    model="gemini-1.5-flash", contents="test content"
                )
                async for _ in stream:
                    break
                del stream
                # The event loop closes collected async generators in tasks it schedules.
                for _ in range(10):
                    if released:
                        break
                    gc.collect()
                    await asyncio.sleep(0)
                # Checked before `asyncio.run` returns, because it closes leftover async generators.
                assert released == [True]

            asyncio.run(_abandon_stream())
        else:
            stream = client.models.generate_content_stream(
                model="gemini-1.5-flash", contents="test content"
            )
            for _ in stream:
                break
            del stream
            gc.collect()
            assert released == [True]

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"


def test_async_generate_content_stream_cancelled():
    # Cancelling the consuming task (e.g. a client disconnecting from a streaming endpoint) must
    # end the spans with an error right away, not leave them open until garbage collection.
    async def _generate_content_stream(self, model, contents, config):
        async def _stream():
            yield _STREAM_CHUNKS[0]
            await asyncio.Event().wait()

        return _stream()

    async def _cancel_stream():
        client = genai.Client(api_key="dummy")
        stream = await client.aio.models.generate_content_stream(
            model="gemini-1.5-flash", contents="test content"
        )
        first_chunk_received = asyncio.Event()

        async def _consume():
            async for _ in stream:
                first_chunk_received.set()

        task = asyncio.create_task(_consume())
        await first_chunk_received.wait()
        task.cancel()
        await asyncio.wait([task])
        assert task.cancelled()
        # `stream` is still referenced, so the trace is exported only if cancellation ended it.
        assert len(get_traces()) == 1

    with patch(
        "google.genai.models.AsyncModels._generate_content_stream", new=_generate_content_stream
    ):
        mlflow.gemini.autolog()
        asyncio.run(_cancel_stream())

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    assert len(traces[0].data.spans) == 2
    for span in traces[0].data.spans:
        assert span.status.status_code == "ERROR"
        assert span.status.description == "CancelledError: "
        assert span.events[-1].name == "exception"
        assert span.events[-1].attributes["exception.type"] == "CancelledError"
        assert span.outputs["candidates"][0]["content"]["parts"] == [
            _STREAM_CHUNKS[0].candidates[0].content.parts[0].model_dump()
        ]


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
        assert span.model_name == "gemini-1.5-flash"

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
        assert span.model_name == "text-embedding-004"

        mlflow.gemini.autolog(disable=True)
        client.models.embed_content(model="text-embedding-004", content="Hello World")

        # No new trace should be created
        traces = get_traces()
        assert len(traces) == 1


def test_generate_content_cached_tokens(is_async, mock_litellm_cost):
    cached_response = _generate_content_response(_CONTENT, _USER_METADATA_WITH_CACHE)

    if is_async:

        async def _generate_content(self, model, contents, config):
            return cached_response

    else:

        def _generate_content(self, model, contents, config):
            return cached_response

    cls = "AsyncModels" if is_async else "Models"
    with patch(f"google.genai.models.{cls}._generate_content", new=_generate_content):
        mlflow.gemini.autolog()
        _call_generate_content(is_async, "test content")

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 50,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 70,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 30,
    }


def test_generate_content_thoughts_tokens(is_async, mock_litellm_cost):
    thinking_response = _generate_content_response(
        _CONTENT,
        {
            "prompt_token_count": 10,
            "candidates_token_count": 2,
            "thoughts_token_count": 300,
            "total_token_count": 312,
        },
    )

    if is_async:

        async def _generate_content(self, model, contents, config):
            return thinking_response

    else:

        def _generate_content(self, model, contents, config):
            return thinking_response

    cls = "AsyncModels" if is_async else "Models"
    with patch(f"google.genai.models.{cls}._generate_content", new=_generate_content):
        mlflow.gemini.autolog()
        _call_generate_content(is_async, "test content")

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 302,
        TokenUsageKey.TOTAL_TOKENS: 312,
    }


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_input_tokens"),
    [
        # Live gemini-3.5-flash-lite response with the URL context tool
        (27, 2447),
        (None, 2420),
    ],
)
def test_generate_content_tool_use_prompt_tokens(
    is_async, mock_litellm_cost, prompt_tokens, expected_input_tokens
):
    tool_use_response = _generate_content_response(
        _CONTENT,
        {
            "prompt_token_count": prompt_tokens,
            "tool_use_prompt_token_count": 2420,
            "candidates_token_count": 73,
            "total_token_count": expected_input_tokens + 73,
        },
    )

    if is_async:

        async def _generate_content(self, model, contents, config):
            return tool_use_response

    else:

        def _generate_content(self, model, contents, config):
            return tool_use_response

    cls = "AsyncModels" if is_async else "Models"
    with patch(f"google.genai.models.{cls}._generate_content", new=_generate_content):
        mlflow.gemini.autolog()
        _call_generate_content(is_async, "test content")

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: expected_input_tokens,
        TokenUsageKey.OUTPUT_TOKENS: 73,
        TokenUsageKey.TOTAL_TOKENS: expected_input_tokens + 73,
    }

    if not IS_TRACING_SDK_ONLY:
        # input tokens * 1.0 + output tokens * 2.0
        assert span.llm_cost == {
            "input_cost": float(expected_input_tokens),
            "output_cost": 146.0,
            "total_cost": expected_input_tokens + 146.0,
        }


@pytest.fixture(params=[False, True], ids=["non_stream", "stream"])
def is_stream(request):
    return request.param


def _request_method(is_stream: bool) -> str:
    return "_generate_content_stream" if is_stream else "_generate_content"


def _patch_model_request(is_async: bool, is_stream: bool, next_response):
    # `next_response(config)` returns the response for one model request. A streaming request
    # yields it as a single chunk.
    if is_stream and is_async:

        async def request(self, model, contents, config):
            response = next_response(config)

            async def _stream():
                yield response

            return _stream()

    elif is_stream:

        def request(self, model, contents, config):
            yield next_response(config)

    elif is_async:

        async def request(self, model, contents, config):
            return next_response(config)

    else:

        def request(self, model, contents, config):
            return next_response(config)

    method = _request_method(is_stream)
    # autolog decides whether to inject headers from the patched method's name
    request.__name__ = method
    cls = "AsyncModels" if is_async else "Models"
    return patch(f"google.genai.models.{cls}.{method}", new=request)


def _capture_configs(captured_configs: list[object]):
    def next_response(config):
        captured_configs.append(config)
        return _DUMMY_GENERATE_CONTENT_RESPONSE

    return next_response


def _call_model(is_async: bool, is_stream: bool, contents: str = "test content", config=None):
    if is_stream:
        return _call_generate_content_stream(is_async, contents, config=config)
    return _call_generate_content(is_async, contents, config=config)


def _call_chat(is_async: bool, is_stream: bool, contents: str, config):
    client = genai.Client(api_key="dummy")
    if is_async:

        async def _call():
            chat = client.aio.chats.create(model="gemini-1.5-flash", config=config)
            if is_stream:
                return [chunk async for chunk in await chat.send_message_stream(contents)]
            return await chat.send_message(contents)

        return asyncio.run(_call())
    chat = client.chats.create(model="gemini-1.5-flash", config=config)
    if is_stream:
        return list(chat.send_message_stream(contents))
    return chat.send_message(contents)


def _request_headers(config) -> dict[str, str]:
    if isinstance(config, dict):
        http_options = config.get("http_options")
    else:
        http_options = getattr(config, "http_options", None)
    if isinstance(http_options, dict):
        return http_options.get("headers") or {}
    return getattr(http_options, "headers", None) or {}


def _assert_traceparent_identifies(traceparent: str, span):
    match = re.fullmatch(r"00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}", traceparent)
    assert match
    assert match.groups() == (span.trace_id.removeprefix("tr-"), span.span_id)


def _request_spans(trace, is_async: bool, is_stream: bool):
    cls = "AsyncModels" if is_async else "Models"
    name = f"{cls}.{_request_method(is_stream)}"
    return [span for span in trace.data.spans if span.name == name]


def test_tracing_headers_injected_in_config(is_async, is_stream):
    captured_configs = []
    with _patch_model_request(is_async, is_stream, _capture_configs(captured_configs)):
        mlflow.gemini.autolog()
        _call_model(is_async, is_stream, config={"temperature": 0.5})

    [trace] = get_traces()
    [config] = captured_configs
    [request_span] = _request_spans(trace, is_async, is_stream)
    _assert_traceparent_identifies(_request_headers(config)["traceparent"], request_span)


def test_tracing_headers_preserve_existing_config_headers(is_async, is_stream):
    captured_configs = []
    with _patch_model_request(is_async, is_stream, _capture_configs(captured_configs)):
        mlflow.gemini.autolog()
        _call_model(
            is_async,
            is_stream,
            config={
                "temperature": 0.5,
                "http_options": {"headers": {"X-Custom": "my-value"}},
            },
        )

    [config] = captured_configs
    headers = _request_headers(config)
    assert "traceparent" in headers
    assert headers["X-Custom"] == "my-value"


def test_tracing_headers_injected_when_config_is_none(is_async, is_stream):
    captured_configs = []
    with _patch_model_request(is_async, is_stream, _capture_configs(captured_configs)):
        mlflow.gemini.autolog()
        _call_model(is_async, is_stream)

    [trace] = get_traces()
    [config] = captured_configs
    [request_span] = _request_spans(trace, is_async, is_stream)
    _assert_traceparent_identifies(_request_headers(config)["traceparent"], request_span)

    # The injected header is an implementation detail and must not leak into span inputs
    for span in trace.data.spans:
        if (config_input := span.inputs.get("config")) is not None:
            assert "traceparent" not in _request_headers(config_input)


@pytest.mark.parametrize(
    "config",
    [
        {"temperature": 0.5, "http_options": {"headers": {"X-Custom": "my-value"}}},
        {"temperature": 0.5, "http_options": genai.types.HttpOptions(headers={"X-Custom": "a"})},
        genai.types.GenerateContentConfig(temperature=0.5),
        genai.types.GenerateContentConfig(
            temperature=0.5, http_options=genai.types.HttpOptions(headers={"X-Custom": "a"})
        ),
    ],
    ids=["dict", "dict_with_http_options", "object", "object_with_http_options"],
)
def test_tracing_headers_with_reused_config(is_async, is_stream, config):
    # A traceparent written into the caller's config would be sent again by the next call that
    # reuses the config, pointing at the previous trace.
    config = copy.deepcopy(config)
    captured_configs = []
    with _patch_model_request(is_async, is_stream, _capture_configs(captured_configs)):
        mlflow.gemini.autolog()
        _call_model(is_async, is_stream, config=config)
        _call_model(is_async, is_stream, config=config)

    traces = {trace.info.trace_id.removeprefix("tr-"): trace for trace in get_traces()}
    traceparents = [_request_headers(c)["traceparent"] for c in captured_configs]
    assert len(traces) == len(traceparents) == 2
    # Each call must point at its own trace
    assert {traceparent.split("-")[1] for traceparent in traceparents} == traces.keys()
    for traceparent in traceparents:
        trace = traces[traceparent.split("-")[1]]
        [request_span] = _request_spans(trace, is_async, is_stream)
        _assert_traceparent_identifies(traceparent, request_span)
    assert "traceparent" not in _request_headers(config)


@pytest.mark.parametrize("api", ["models", "chat"])
def test_tracing_headers_identify_each_request(is_async, is_stream, api):
    # Automatic function calling sends one model request per round and reuses one config across
    # rounds, so each request must carry its own span rather than an earlier round's.
    traceparents = []

    def _next_response(config):
        traceparents.append(_request_headers(config)["traceparent"])
        return _FUNCTION_CALL_CHUNK if len(traceparents) == 1 else _FUNCTION_ANSWER_CHUNK

    config = genai.types.GenerateContentConfig(tools=[multiply])
    with _patch_model_request(is_async, is_stream, _next_response):
        mlflow.gemini.autolog()
        if api == "models":
            _call_model(is_async, is_stream, "What is 57 * 44?", config=config)
        else:
            _call_chat(is_async, is_stream, "What is 57 * 44?", config)

    [trace] = get_traces()
    request_spans = {span.span_id: span for span in _request_spans(trace, is_async, is_stream)}
    assert len(request_spans) == len(traceparents) == 2
    # Each request must point at a different request span
    assert {traceparent.split("-")[2] for traceparent in traceparents} == request_spans.keys()
    for traceparent in traceparents:
        _assert_traceparent_identifies(traceparent, request_spans[traceparent.split("-")[2]])
    # A traceparent written into the caller's config would be resent on its next call
    assert "traceparent" not in _request_headers(config)
