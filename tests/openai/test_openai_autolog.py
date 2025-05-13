import json
from unittest import mock

import httpx
import openai
import pytest
from openai._types import NOT_GIVEN, NotGiven
from packaging.version import Version
from pydantic import BaseModel

import mlflow
from mlflow.entities.span import SpanType
from mlflow.exceptions import MlflowException
from mlflow.openai.utils.chat_schema import _parse_tools
from mlflow.tracing.constant import STREAM_CHUNK_EVENT_VALUE_KEY, SpanAttributeKey, TraceMetadataKey

from tests.openai.mock_openai import EMPTY_CHOICES
from tests.tracing.helper import get_traces, skip_when_testing_trace_sdk

MOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


@pytest.fixture(params=[True, False], ids=["sync", "async"])
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


@pytest.mark.asyncio
async def test_chat_completions_autolog(client):
    mlflow.openai.autolog()

    messages = [{"role": "user", "content": "test"}]
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0,
    )

    if client._is_async:
        await response

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == {"messages": messages, "model": "gpt-4o-mini", "temperature": 0}
    assert span.outputs["id"] == "chatcmpl-123"
    assert span.attributes["model"] == "gpt-4o-mini"
    assert span.attributes["temperature"] == 0

    assert TraceMetadataKey.SOURCE_RUN not in trace.info.request_metadata


@pytest.mark.asyncio
async def test_chat_completions_autolog_under_current_active_span(client):
    # If a user have an active span, the autologging should create a child span under it.
    mlflow.openai.autolog()

    messages = [{"role": "user", "content": "test"}]
    with mlflow.start_span(name="parent"):
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            temperature=0,
        )

        if client._is_async:
            await response

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 2
    parent_span = trace.data.spans[0]
    assert parent_span.name == "parent"
    child_span = trace.data.spans[1]
    assert child_span.name == "AsyncCompletions" if client._is_async else "Completions"
    assert child_span.inputs == {"messages": messages, "model": "gpt-4o-mini", "temperature": 0}
    assert child_span.outputs["id"] == "chatcmpl-123"
    assert child_span.parent_id == parent_span.span_id


@pytest.mark.asyncio
async def test_chat_completions_autolog_streaming(client):
    mlflow.openai.autolog()

    messages = [{"role": "user", "content": "test"}]
    stream = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0,
        stream=True,
    )

    if client._is_async:
        async for _ in await stream:
            pass
    else:
        for _ in stream:
            pass

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == {
        "messages": messages,
        "model": "gpt-4o-mini",
        "temperature": 0,
        "stream": True,
    }
    assert span.outputs == "Hello world"  # aggregated string of streaming response

    stream_event_data = trace.data.spans[0].events
    assert stream_event_data[0].name == "mlflow.chunk.item.0"
    chunk_1 = json.loads(stream_event_data[0].attributes[STREAM_CHUNK_EVENT_VALUE_KEY])
    assert chunk_1["id"] == "chatcmpl-123"
    assert chunk_1["choices"][0]["delta"]["content"] == "Hello"
    assert stream_event_data[1].name == "mlflow.chunk.item.1"
    chunk_2 = json.loads(stream_event_data[1].attributes[STREAM_CHUNK_EVENT_VALUE_KEY])
    assert chunk_2["id"] == "chatcmpl-123"
    assert chunk_2["choices"][0]["delta"]["content"] == " world"


@pytest.mark.asyncio
async def test_chat_completions_autolog_tracing_error(client):
    mlflow.openai.autolog()
    messages = [{"role": "user", "content": "test"}]
    with pytest.raises(openai.UnprocessableEntityError, match="Input should be less"):  # noqa: PT012
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            temperature=5.0,
        )

        if client._is_async:
            await response

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "ERROR"

    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.name == "AsyncCompletions" if client._is_async else "Completions"
    assert span.inputs["messages"][0]["content"] == "test"
    assert span.outputs is None

    assert span.events[0].name == "exception"
    assert span.events[0].attributes["exception.type"] == "UnprocessableEntityError"


@pytest.mark.asyncio
async def test_chat_completions_autolog_tracing_error_with_parent_span(client):
    mlflow.openai.autolog()

    if client._is_async:

        @mlflow.trace
        async def create_completions(text: str) -> str:
            try:
                response = await client.chat.completions.create(
                    messages=[{"role": "user", "content": text}],
                    model="gpt-4o-mini",
                    temperature=5.0,
                )
                return response.choices[0].delta.content
            except openai.OpenAIError as e:
                raise MlflowException("Failed to create completions") from e

        with pytest.raises(MlflowException, match="Failed to create completions"):
            await create_completions("test")

    else:

        @mlflow.trace
        def create_completions(text: str) -> str:
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": text}],
                    model="gpt-4o-mini",
                    temperature=5.0,
                )
                return response.choices[0].delta.content
            except openai.OpenAIError as e:
                raise MlflowException("Failed to create completions") from e

        with pytest.raises(MlflowException, match="Failed to create completions"):
            create_completions("test")

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "ERROR"

    assert len(trace.data.spans) == 2
    parent_span = trace.data.spans[0]
    assert parent_span.name == "create_completions"
    assert parent_span.inputs == {"text": "test"}
    assert parent_span.outputs is None
    assert parent_span.status.status_code == "ERROR"
    assert parent_span.events[0].name == "exception"
    assert parent_span.events[0].attributes["exception.type"] == "mlflow.exceptions.MlflowException"
    assert parent_span.events[0].attributes["exception.message"] == "Failed to create completions"

    child_span = trace.data.spans[1]
    assert child_span.name == "AsyncCompletions" if client._is_async else "Completions"
    assert child_span.inputs["messages"][0]["content"] == "test"
    assert child_span.outputs is None
    assert child_span.status.status_code == "ERROR"
    assert child_span.events[0].name == "exception"
    assert child_span.events[0].attributes["exception.type"] == "UnprocessableEntityError"


@pytest.mark.asyncio
async def test_chat_completions_streaming_empty_choices(client):
    mlflow.openai.autolog()
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": EMPTY_CHOICES}],
        model="gpt-4o-mini",
        stream=True,
    )

    if client._is_async:
        chunks = []
        async for chunk in await stream:
            chunks.append(chunk)
    else:
        chunks = list(stream)

    # Ensure the stream has a chunk with empty choices
    assert chunks[0].choices == []

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "OK"


@pytest.mark.asyncio
async def test_completions_autolog(client):
    mlflow.openai.autolog()

    response = client.completions.create(
        prompt="test",
        model="gpt-4o-mini",
        temperature=0,
    )

    if client._is_async:
        await response

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.span_type == SpanType.LLM
    assert span.inputs == {"prompt": "test", "model": "gpt-4o-mini", "temperature": 0}
    assert span.outputs["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"

    assert TraceMetadataKey.SOURCE_RUN not in trace.info.request_metadata


@pytest.mark.asyncio
async def test_completions_autolog_streaming_empty_choices(client):
    mlflow.openai.autolog()
    stream = client.completions.create(
        prompt=EMPTY_CHOICES,
        model="gpt-4o-mini",
        stream=True,
    )

    if client._is_async:
        chunks = []
        async for chunk in await stream:
            chunks.append(chunk)
    else:
        chunks = list(stream)

    # Ensure the stream has a chunk with empty choices
    assert chunks[0].choices == []

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace.info.status == "OK"


@pytest.mark.asyncio
async def test_completions_autolog_streaming(client):
    mlflow.openai.autolog()

    stream = client.completions.create(
        prompt="test",
        model="gpt-4o-mini",
        temperature=0,
        stream=True,
    )
    if client._is_async:
        async for _ in await stream:
            pass
    else:
        for _ in stream:
            pass

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.span_type == SpanType.LLM
    assert span.inputs == {
        "prompt": "test",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "stream": True,
    }
    assert span.outputs == "Hello world"  # aggregated string of streaming response

    stream_event_data = trace.data.spans[0].events

    assert stream_event_data[0].name == "mlflow.chunk.item.0"
    chunk_1 = json.loads(stream_event_data[0].attributes[STREAM_CHUNK_EVENT_VALUE_KEY])
    assert chunk_1["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"
    assert chunk_1["choices"][0]["text"] == "Hello"
    assert stream_event_data[1].name == "mlflow.chunk.item.1"
    chunk_2 = json.loads(stream_event_data[1].attributes[STREAM_CHUNK_EVENT_VALUE_KEY])
    assert chunk_2["id"] == "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"
    assert chunk_2["choices"][0]["text"] == " world"


@pytest.mark.asyncio
async def test_embeddings_autolog(client):
    mlflow.openai.autolog()

    response = client.embeddings.create(
        input="test",
        model="text-embedding-ada-002",
    )

    if client._is_async:
        await response

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None
    assert trace.info.status == "OK"
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.span_type == SpanType.EMBEDDING
    assert span.inputs == {"input": "test", "model": "text-embedding-ada-002"}
    assert span.outputs["data"][0]["embedding"] == list(range(1536))

    assert TraceMetadataKey.SOURCE_RUN not in trace.info.request_metadata


@skip_when_testing_trace_sdk
@pytest.mark.asyncio
async def test_autolog_use_active_run_id(client):
    mlflow.openai.autolog()

    messages = [{"role": "user", "content": "test"}]

    async def _call_create():
        response = client.chat.completions.create(messages=messages, model="gpt-4o-mini")
        if client._is_async:
            await response
        return response

    with mlflow.start_run() as run_1:
        await _call_create()

    with mlflow.start_run() as run_2:
        await _call_create()
        await _call_create()

    with mlflow.start_run() as run_3:
        mlflow.openai.autolog()
        await _call_create()

    traces = get_traces()[::-1]  # reverse order to sort by timestamp in ascending order
    assert len(traces) == 4

    assert traces[0].info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_1.info.run_id
    assert traces[1].info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_2.info.run_id
    assert traces[2].info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_2.info.run_id
    assert traces[3].info.request_metadata[TraceMetadataKey.SOURCE_RUN] == run_3.info.run_id


@pytest.mark.asyncio
async def test_autolog_raw_response(client):
    mlflow.openai.autolog()

    messages = [{"role": "user", "content": "test"}]

    resp = client.chat.completions.with_raw_response.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=MOCK_TOOLS,
    )

    if client._is_async:
        resp = await resp

    resp = resp.parse()  # ensure the raw response is returned

    assert resp.choices[0].message.content == '[{"role": "user", "content": "test"}]'
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL
    assert isinstance(span.outputs, dict)
    assert (
        span.outputs["choices"][0]["message"]["content"] == '[{"role": "user", "content": "test"}]'
    )
    assert span.attributes[SpanAttributeKey.CHAT_MESSAGES] == (
        messages + [{"role": "assistant", "content": '[{"role": "user", "content": "test"}]'}]
    )
    assert span.attributes[SpanAttributeKey.CHAT_TOOLS] == MOCK_TOOLS


@pytest.mark.asyncio
async def test_autolog_raw_response_stream(client):
    mlflow.openai.autolog()

    messages = [{"role": "user", "content": "test"}]

    resp = client.chat.completions.with_raw_response.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=MOCK_TOOLS,
        stream=True,
    )

    if client._is_async:
        resp = await resp

    resp = resp.parse()  # ensure the raw response is returned

    if client._is_async:
        chunks = [c.choices[0].delta.content async for c in resp]
    else:
        chunks = [c.choices[0].delta.content for c in resp]
    assert chunks == ["Hello", " world"]
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.outputs == "Hello world"
    assert span.attributes[SpanAttributeKey.CHAT_MESSAGES] == (
        messages + [{"role": "assistant", "content": "Hello world"}]
    )
    assert span.attributes[SpanAttributeKey.CHAT_TOOLS] == MOCK_TOOLS


@pytest.mark.skipif(
    Version(openai.__version__) < Version("1.40"), reason="Requires OpenAI SDK >= 1.40"
)
@pytest.mark.asyncio
async def test_response_format(client):
    mlflow.openai.autolog()

    class Person(BaseModel):
        name: str
        age: int

    mock_response = {
        "id": "chatcmpl-Ax4UAd5xf32KjgLkS1SEEY9oorI9m",
        "object": "chat.completion",
        "created": 1738641958,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"name":"Angelo","age":42}',
                    "refusal": None,
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 68,
            "completion_tokens": 11,
            "total_tokens": 79,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": "fp_50cad350e4",
    }

    if client._is_async:
        patch_target = "httpx.AsyncClient.send"

        async def send_patch(self, request, *args, **kwargs):
            return httpx.Response(
                status_code=200,
                request=request,
                json=mock_response,
            )
    else:
        patch_target = "httpx.Client.send"

        def send_patch(self, request, *args, **kwargs):
            return httpx.Response(
                status_code=200,
                request=request,
                json=mock_response,
            )

    with mock.patch(patch_target, send_patch):
        response = client.beta.chat.completions.parse(
            messages=[
                {"role": "system", "content": "Extract info from text"},
                {"role": "user", "content": "I am Angelo and I am 42."},
            ],
            model="gpt-4o",
            temperature=0,
            response_format=Person,
        )

        if client._is_async:
            response = await response

    assert response.choices[0].message.parsed == Person(name="Angelo", age=42)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert len(trace.data.spans) == 1
    span = trace.data.spans[0]
    assert span.outputs["choices"][0]["message"]["content"] == '{"name":"Angelo","age":42}'
    assert span.span_type == SpanType.CHAT_MODEL


@skip_when_testing_trace_sdk
@pytest.mark.asyncio
async def test_autolog_link_traces_to_loaded_model_chat_completions(client):
    mlflow.openai.autolog()

    temperatures = [temp / 10 for temp in range(1, 5)]
    model_infos = []
    for temp in temperatures:
        with mlflow.start_run():
            model_infos.append(
                mlflow.openai.log_model(
                    "gpt-4o-mini",
                    "chat.completions",
                    name="model",
                    temperature=temp,
                    messages=[{"role": "system", "content": "You are an MLflow expert."}],
                )
            )

    for model_info in model_infos:
        model_dict = mlflow.openai.load_model(model_info.model_uri)
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": f"test {model_info.model_id}"}],
            model=model_dict["model"],
            temperature=model_dict["temperature"],
        )
        if client._is_async:
            await resp

    traces = get_traces()
    assert len(traces) == len(temperatures)
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[SpanAttributeKey.MODEL_ID]
        assert model_id is not None
        assert span.inputs["messages"][0]["content"] == f"test {model_id}"


@skip_when_testing_trace_sdk
@pytest.mark.asyncio
async def test_autolog_link_traces_to_loaded_model_completions(client):
    mlflow.openai.autolog()

    temperatures = [temp / 10 for temp in range(1, 5)]
    model_infos = []
    for temp in temperatures:
        with mlflow.start_run():
            model_infos.append(
                mlflow.openai.log_model(
                    "gpt-4o-mini",
                    "completions",
                    name="model",
                    temperature=temp,
                    prompt="Say {text}",
                )
            )

    for model_info in model_infos:
        model_dict = mlflow.openai.load_model(model_info.model_uri)
        resp = client.completions.create(
            prompt=f"test {model_info.model_id}",
            model=model_dict["model"],
            temperature=model_dict["temperature"],
        )
        if client._is_async:
            await resp

    traces = get_traces()
    assert len(traces) == len(temperatures)
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[SpanAttributeKey.MODEL_ID]
        assert model_id is not None
        assert span.inputs["prompt"] == f"test {model_id}"


@skip_when_testing_trace_sdk
@pytest.mark.asyncio
async def test_autolog_link_traces_to_loaded_model_embeddings(client):
    mlflow.openai.autolog()

    encoding_formats = ["float", "base64"]
    model_infos = []
    for encoding_format in encoding_formats:
        with mlflow.start_run():
            model_infos.append(
                mlflow.openai.log_model(
                    "text-embedding-ada-002",
                    "embeddings",
                    name="model",
                    encoding_format=encoding_format,
                )
            )

    for model_info in model_infos:
        model_dict = mlflow.openai.load_model(model_info.model_uri)
        resp = client.embeddings.create(
            input=f"test {model_info.model_id}",
            model=model_dict["model"],
            encoding_format=model_dict["encoding_format"],
        )
        if client._is_async:
            await resp

    traces = get_traces()
    assert len(traces) == len(encoding_formats)
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[SpanAttributeKey.MODEL_ID]
        assert model_id is not None
        assert span.inputs["input"] == f"test {model_id}"


@skip_when_testing_trace_sdk
def test_autolog_link_traces_to_loaded_model_embeddings_pyfunc(monkeypatch, mock_openai):
    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_BASE": mock_openai,
        }
    )

    mlflow.openai.autolog()

    encoding_formats = ["float", "base64"]
    model_infos = []
    for encoding_format in encoding_formats:
        with mlflow.start_run():
            model_infos.append(
                mlflow.openai.log_model(
                    "text-embedding-ada-002",
                    "embeddings",
                    name="model",
                    encoding_format=encoding_format,
                )
            )

    for model_info in model_infos:
        pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
        assert mlflow.get_active_model_id() == model_info.model_id
        pyfunc_model.predict(f"test {model_info.model_id}")

    traces = get_traces()
    assert len(traces) == len(encoding_formats)
    for trace in traces:
        span = trace.data.spans[0]
        model_id = trace.info.request_metadata[SpanAttributeKey.MODEL_ID]
        assert model_id is not None
        assert span.inputs["input"] == [f"test {model_id}"]


@skip_when_testing_trace_sdk
def test_autolog_link_traces_to_active_model(monkeypatch, mock_openai):
    monkeypatch.setenvs(
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_API_BASE": mock_openai,
        }
    )

    model = mlflow.create_external_model(name="test_model")
    mlflow.set_active_model(model_id=model.model_id)
    mlflow.openai.autolog()

    encoding_formats = ["float", "base64"]
    model_infos = []
    for encoding_format in encoding_formats:
        with mlflow.start_run():
            model_infos.append(
                mlflow.openai.log_model(
                    "text-embedding-ada-002",
                    "embeddings",
                    name="model",
                    encoding_format=encoding_format,
                )
            )

    for model_info in model_infos:
        pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
        pyfunc_model.predict(model_info.model_id)

    traces = get_traces()
    assert len(traces) == len(encoding_formats)
    for trace in traces:
        span = trace.data.spans[0]
        assert trace.info.request_metadata[SpanAttributeKey.MODEL_ID] == model.model_id
        logged_model_id = span.inputs["input"][0]
        assert logged_model_id != model.model_id


@pytest.mark.parametrize(
    "sentinel",
    [
        NOT_GIVEN,
        NotGiven(),
    ],
)
def test_parse_tools_handles_openai_not_given_sentinel(sentinel):
    assert _parse_tools({"tools": sentinel}) == []

@skip_when_testing_trace_sdk
@pytest.mark.asyncio
async def test_model_loading_set_active_model_id_without_fetching_logged_model(client):
    mlflow.openai.autolog()

    model_info = mlflow.openai.log_model(
        "gpt-4o-mini",
        "chat.completions",
        name="model",
        temperature=0.9,
        messages=[{"role": "system", "content": "You are an MLflow expert."}],
    )

    with mock.patch("mlflow.get_logged_model", side_effect=Exception("get_logged_model failed")):
        model_dict = mlflow.openai.load_model(model_info.model_uri)
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": f"test {model_info.model_id}"}],
        model=model_dict["model"],
        temperature=model_dict["temperature"],
    )
    if client._is_async:
        await resp

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    model_id = traces[0].info.request_metadata[SpanAttributeKey.MODEL_ID]
    assert model_id is not None
    assert span.inputs["messages"][0]["content"] == f"test {model_id}"
