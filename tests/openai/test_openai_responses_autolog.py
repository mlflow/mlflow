import openai
import pytest
from packaging.version import Version

import mlflow
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

from tests.tracing.helper import get_traces

if Version(openai.__version__) < Version("1.66.00"):
    pytest.skip(
        "OpenAI < 1.66.0 does not support the Responses API.",
        allow_module_level=True,
    )


@pytest.fixture(params=[True, False], ids=["sync", "async"])
def client(request, monkeypatch, mock_openai):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_BASE", mock_openai)
    if request.param:
        client = openai.OpenAI(api_key="test", base_url=mock_openai)
        client._is_async = False
        return client
    else:
        client = openai.AsyncOpenAI(api_key="test", base_url=mock_openai)
        client._is_async = True
        return client


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

    # Token usage should be aggregated correctly
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 36,
        TokenUsageKey.OUTPUT_TOKENS: 87,
        TokenUsageKey.TOTAL_TOKENS: 123,
    }


@pytest.mark.asyncio
async def test_responses_image_input_autolog(client):
    mlflow.openai.autolog()

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


@pytest.mark.asyncio
async def test_responses_web_search_autolog(client):
    mlflow.openai.autolog()

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
    assert span.attributes[SpanAttributeKey.CHAT_TOOLS] == [
        {"type": "function", "function": {"name": "web_search_preview"}}
    ]


@pytest.mark.asyncio
async def test_responses_file_search_autolog(client):
    mlflow.openai.autolog()

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
    assert llm_span_1.attributes[SpanAttributeKey.CHAT_TOOLS] == [
        {"type": "function", "function": {"name": "computer_use_preview"}}
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
    assert span.attributes[SpanAttributeKey.CHAT_TOOLS] == [
        {"type": "function", "function": {k: v for k, v in tools[0].items() if k != "type"}}
    ]
    assert span.attributes[SpanAttributeKey.MESSAGE_FORMAT] == "openai"


@pytest.mark.asyncio
async def test_responses_stream_autolog(client):
    mlflow.openai.autolog()

    response = client.responses.create(
        input="Hello",
        model="gpt-4o",
        stream=True,
    )

    if client._is_async:
        async for _ in await response:
            pass
    else:
        for _ in response:
            pass

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.outputs["id"] == "responses-123"
    # "logprobs" is only returned from certain version of OpenAI SDK
    span.outputs["output"][0]["content"][0].pop("logprobs", None)
    assert span.outputs["output"][0]["content"] == [
        {
            "text": "Dummy output",
            "annotations": None,
            "type": "output_text",
        }
    ]
    assert span.attributes["model"] == "gpt-4o"
    assert span.attributes["stream"] is True

    # Token usage should be aggregated correctly
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 36,
        TokenUsageKey.OUTPUT_TOKENS: 87,
        TokenUsageKey.TOTAL_TOKENS: 123,
    }
