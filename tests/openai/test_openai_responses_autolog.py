import contextlib
from typing import Any
from unittest import mock

import httpx
import openai
import pytest

import mlflow
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces


@pytest.fixture(params=[True])  # , False], ids=["sync", "async"])
def client(request, monkeypatch, mock_openai):
    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_BASE": mock_openai,
        }
    )
    if request.param:
        client = openai.OpenAI(api_key="test", base_url=mock_openai)
        client._is_async = False
        return client
    else:
        client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)
        client._is_async = True
        return client


def _get_responses_payload(outputs, tools=None):
    return {
        "id": "responses-123",
        "object": "response",
        "created": 1589478378,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "max_output_tokens": None,
        "model": "gpt-4o",
        "output": outputs,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "generate_summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": tools or [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 36,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 87,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 123,
        },
        "user": None,
        "metadata": {},
    }


@contextlib.contextmanager
def mock_openai_responses(responses: list[Any]):
    assert isinstance(responses, list)

    cloned = iter(responses.copy())

    def send(self, request, *args, **kwargs):
        return httpx.Response(
            status_code=200,
            request=request,
            json=next(cloned),
        )

    async def async_send(self, request, *args, **kwargs):
        return httpx.Response(
            status_code=200,
            request=request,
            json=next(cloned),
        )

    with mock.patch("httpx.Client.send", send), mock.patch("httpx.AsyncClient.send", async_send):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "_input",
    [
        "Hello",
        [{"role": "user", "content": "Hello"}],
    ],
)
async def test_responses_autolog(client, _input):
    mlflow.openai.autolog()

    dummy_response = _get_responses_payload(
        outputs=[
            {
                "type": "message",
                "id": "test",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Hello world",
                    }
                ],
            }
        ]
    )
    with mock_openai_responses([dummy_response]):
        response = client.responses.create(
            input=_input,
            model="gpt-4o",
            temperature=0,
        )

        if client._is_async:
            await response

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == {"input": _input, "model": "gpt-4o", "temperature": 0}
    assert span.outputs["id"] == "responses-123"
    assert span.attributes["model"] == "gpt-4o"
    assert span.attributes["temperature"] == 0
    assert span.attributes[SpanAttributeKey.CHAT_MESSAGES] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello world"}]},
    ]


@pytest.mark.asyncio
async def test_responses_image_input_autolog(client):
    mlflow.openai.autolog()

    dummy_response = _get_responses_payload(
        outputs=[
            {
                "type": "message",
                "id": "test",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "This is an apple",
                    }
                ],
            }
        ]
    )
    with mock_openai_responses([dummy_response]):
        response = client.responses.create(
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is in this image?"},
                        {
                            "type": "input_image",
                            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        },
                    ],
                }
            ],
            model="gpt-4o",
            temperature=0,
        )

        if client._is_async:
            await response

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs["model"] == "gpt-4o"
    assert span.outputs["id"] == "responses-123"
    assert span.attributes[SpanAttributeKey.CHAT_MESSAGES] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        "detail": None,
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "This is an apple"}],
        },
    ]


@pytest.mark.asyncio
async def test_responses_web_search_autolog(client):
    mlflow.openai.autolog()

    dummy_response = _get_responses_payload(
        outputs=[
            {"type": "web_search_call", "id": "tool_call_1", "status": "completed"},
            {
                "type": "message",
                "id": "msg",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "As of today, March 9, 2025, one notable positive news story...",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "start_index": 442,
                                "end_index": 557,
                                "url": "https://.../?utm_source=chatgpt.com",
                                "title": "...",
                            },
                        ],
                    }
                ],
            },
        ],
    )
    with mock_openai_responses([dummy_response]):
        response = client.responses.create(
            model="gpt-4o",
            tools=[{"type": "web_search_preview"}],
            input="What was a positive news story from today?",
        )

        if client._is_async:
            await response

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs["model"] == "gpt-4o"
    assert span.outputs["id"] == "responses-123"
    assert span.attributes[SpanAttributeKey.CHAT_MESSAGES] == [
        {
            "role": "user",
            "content": "What was a positive news story from today?",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tool_call_1",
                    "type": "function",
                    "function": {"name": "web_search_call", "arguments": ""},
                }
            ],
        },
        {
            "role": "tool",
            "content": mock.ANY,
            "tool_call_id": "tool_call_1",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "As of today, March 9, 2025, one notable positive news story...",
                }
            ],
        },
    ]
    assert span.attributes[SpanAttributeKey.CHAT_TOOLS] == [
        {"type": "function", "function": {"name": "web_search_preview"}}
    ]


@pytest.mark.asyncio
async def test_responses_file_search_autolog(client):
    mlflow.openai.autolog()

    dummy_response = _get_responses_payload(
        outputs=[
            {
                "type": "file_search_call",
                "id": "file_search_1",
                "status": "completed",
                "queries": ["attributes of an ancient brown dragon"],
                "results": None,
            },
            {
                "type": "message",
                "id": "file_search_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The attributes of an ancient brown dragon include...",
                        "annotations": [
                            {
                                "type": "file_citation",
                                "index": 320,
                                "file_id": "file-4wDz5b167pAf72nx1h9eiN",
                                "filename": "dragons.pdf",
                            },
                            {
                                "type": "file_citation",
                                "index": 576,
                                "file_id": "file-4wDz5b167pAf72nx1h9eiN",
                                "filename": "dragons.pdf",
                            },
                        ],
                    }
                ],
            },
        ],
    )
    with mock_openai_responses([dummy_response]):
        response = client.responses.create(
            model="gpt-4o",
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": ["vs_1234567890"],
                    "max_num_results": 20,
                }
            ],
            input="What are the attributes of an ancient brown dragon?",
        )

        if client._is_async:
            await response

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs["model"] == "gpt-4o"
    assert span.outputs["id"] == "responses-123"
    assert span.attributes[SpanAttributeKey.CHAT_MESSAGES] == [
        {
            "role": "user",
            "content": "What are the attributes of an ancient brown dragon?",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "file_search_1",
                    "type": "function",
                    "function": {
                        "name": "file_search_call",
                        "arguments": '{"queries": ["attributes of an ancient brown dragon"]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": mock.ANY,
            "tool_call_id": "file_search_1",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "The attributes of an ancient brown dragon include...",
                }
            ],
        },
    ]
    assert span.attributes[SpanAttributeKey.CHAT_TOOLS] == [
        {"type": "function", "function": {"name": "file_search"}}
    ]


@pytest.mark.asyncio
async def test_responses_computer_use_autolog(client):
    mlflow.openai.autolog()

    computer_tool_def = {
        "type": "computer_use_preview",
        "display_width": 1024,
        "display_height": 768,
        "environment": "browser",
    }

    dummy_response = _get_responses_payload(
        outputs=[
            {
                "type": "reasoning",
                "id": "rs_67cc...",
                "summary": [
                    {"type": "summary_text", "text": "Clicking on the browser address bar."}
                ],
            },
            {
                "type": "computer_call",
                "id": "cu_67cc...",
                "call_id": "computer_call_1",
                "action": {"type": "click", "button": "left", "x": 156, "y": 50},
                "pending_safety_checks": [],
                "status": "completed",
            },
        ]
    )

    with mock_openai_responses([dummy_response, dummy_response]):
        with mlflow.start_span(name="openai_computer_use"):
            response = client.responses.create(
                model="computer-use-preview",
                input=[{"role": "user", "content": "Check the latest OpenAI news on bing.com."}],
                tools=[computer_tool_def],
            )

            if client._is_async:
                await response

            # Send the response back to the computer tool
            response = client.responses.create(
                model="computer-use-preview",
                input=[
                    {
                        "call_id": "computer_call_1",
                        "type": "computer_call_output",
                        "output": {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,screenshot_base64",
                        },
                    }
                ],
                tools=[computer_tool_def],
            )

            if client._is_async:
                await response

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 3
    llm_span_1 = traces[0].data.spans[1]
    assert llm_span_1.span_type == SpanType.CHAT_MODEL
    assert llm_span_1.inputs["model"] == "computer-use-preview"
    assert llm_span_1.outputs["id"] == "responses-123"
    assert llm_span_1.attributes[SpanAttributeKey.CHAT_MESSAGES] == [
        {
            "role": "user",
            "content": "Check the latest OpenAI news on bing.com.",
        },
        {
            "role": "assistant",
            "content": "Clicking on the browser address bar.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "computer_call_1",
                    "type": "function",
                    "function": {"name": "computer_call", "arguments": mock.ANY},
                }
            ],
        },
    ]
    assert llm_span_1.attributes[SpanAttributeKey.CHAT_TOOLS] == [
        {"type": "function", "function": {"name": "computer_use_preview"}}
    ]

    llm_span_2 = traces[0].data.spans[2]
    assert llm_span_2.attributes[SpanAttributeKey.CHAT_MESSAGES] == [
        {
            "role": "tool",
            "content": [
                {
                    "image_url": {"url": "data:image/png;base64,screenshot_base64"},
                    "type": "image_url",
                }
            ],
            "tool_call_id": "computer_call_1",
        },
        {
            "role": "assistant",
            "content": "Clicking on the browser address bar.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "computer_call_1",
                    "type": "function",
                    "function": {"name": "computer_call", "arguments": mock.ANY},
                }
            ],
        },
    ]


@pytest.mark.asyncio
async def test_responses_function_calling_autolog(client):
    mlflow.openai.autolog()

    tools = [
        {
            "type": "function",
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        }
    ]

    dummy_response = _get_responses_payload(
        outputs=[
            {
                "type": "function_call",
                "id": "fc_67ca09c6bedc8190a7abfec07b1a1332096610f474011cc0",
                "call_id": "function_call_1",
                "name": "get_current_weather",
                "arguments": '{"location":"Boston, MA","unit":"celsius"}',
                "status": "completed",
            }
        ],
        tools=tools,
    )

    with mock_openai_responses([dummy_response, dummy_response]):
        response = client.responses.create(
            model="gpt-4o",
            tools=tools,
            input="What is the weather like in Boston today?",
            tool_choice="auto",
        )

        if client._is_async:
            await response

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs["model"] == "gpt-4o"
    assert span.outputs["id"] == "responses-123"
    assert span.attributes[SpanAttributeKey.CHAT_MESSAGES] == [
        {
            "role": "user",
            "content": "What is the weather like in Boston today?",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "function_call_1",
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": '{"location":"Boston, MA","unit":"celsius"}',
                    },
                }
            ],
        },
    ]
    assert span.attributes[SpanAttributeKey.CHAT_TOOLS] == [
        {"type": "function", "function": {k: v for k, v in tools[0].items() if k != "type"}}
    ]
