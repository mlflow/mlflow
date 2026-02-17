import pytest

from mlflow.gateway.schemas.chat import StreamResponsePayload
from mlflow.gateway.tracing_utils import aggregate_chat_stream_chunks, maybe_traced_gateway_call
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.types.chat import ChatChoiceDelta, ChatChunkChoice, ChatUsage, Function, ToolCallDelta


def get_traces():
    return TracingClient().search_traces(locations=[_get_experiment_id()])


@pytest.fixture
def endpoint_config():
    return GatewayEndpointConfig(
        endpoint_id="test-endpoint-id",
        endpoint_name="test-endpoint",
        experiment_id=_get_experiment_id(),
        models=[],
    )


@pytest.fixture
def endpoint_config_no_experiment():
    return GatewayEndpointConfig(
        endpoint_id="test-endpoint-id",
        endpoint_name="test-endpoint",
        experiment_id=None,
        models=[],
    )


async def mock_async_func(payload):
    return {"result": "success", "payload": payload}


def _make_chunk(
    content=None,
    finish_reason=None,
    id="chunk-1",
    model="test-model",
    created=1700000000,
    usage=None,
    tool_calls=None,
    role="assistant",
    choice_index=0,
):
    delta = ChatChoiceDelta(role=role, content=content, tool_calls=tool_calls)
    choice = ChatChunkChoice(index=choice_index, finish_reason=finish_reason, delta=delta)
    return StreamResponsePayload(
        id=id,
        created=created,
        model=model,
        choices=[choice],
        usage=usage,
    )


def test_aggregate_chat_stream_chunks_aggregates_content():
    chunks = [
        _make_chunk(content="Hello"),
        _make_chunk(content=" "),
        _make_chunk(content="world"),
        _make_chunk(content=None, finish_reason="stop"),
    ]
    result = aggregate_chat_stream_chunks(chunks)

    assert result["object"] == "chat.completion"
    assert result["model"] == "test-model"
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert result["choices"][0]["finish_reason"] == "stop"


def test_aggregate_chat_stream_chunks_with_usage():
    usage = ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunks = [
        _make_chunk(content="Hi"),
        _make_chunk(content=None, finish_reason="stop", usage=usage),
    ]
    result = aggregate_chat_stream_chunks(chunks)

    assert result["choices"][0]["message"]["content"] == "Hi"
    assert result["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_aggregate_chat_stream_chunks_empty():
    assert aggregate_chat_stream_chunks([]) is None


def test_aggregate_chat_stream_chunks_defaults_finish_reason():
    chunks = [_make_chunk(content="Hi")]
    result = aggregate_chat_stream_chunks(chunks)

    assert result["choices"][0]["finish_reason"] == "stop"


def test_reduce_chat_stream_chunks_aggregates_tool_calls():
    chunks = [
        # First chunk: tool call id, type, and function name
        _make_chunk(
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    id="call_abc",
                    type="function",
                    function=Function(name="get_weather", arguments=""),
                ),
            ],
        ),
        # Subsequent chunks: argument fragments
        _make_chunk(
            tool_calls=[
                ToolCallDelta(index=0, function=Function(arguments='{"loc')),
            ],
        ),
        _make_chunk(
            tool_calls=[
                ToolCallDelta(index=0, function=Function(arguments='ation": "SF"}')),
            ],
        ),
        _make_chunk(finish_reason="tool_calls"),
    ]
    result = aggregate_chat_stream_chunks(chunks)

    assert result["choices"][0]["message"]["content"] is None
    assert result["choices"][0]["finish_reason"] == "tool_calls"

    tool_calls = result["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_abc"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert tool_calls[0]["function"]["arguments"] == '{"location": "SF"}'


def test_reduce_chat_stream_chunks_derives_role_from_delta():
    chunks = [
        _make_chunk(role="developer", content="Hello"),
        _make_chunk(role=None, content=" world"),
        _make_chunk(role=None, finish_reason="stop"),
    ]
    result = aggregate_chat_stream_chunks(chunks)

    assert result["choices"][0]["message"]["role"] == "developer"


def test_reduce_chat_stream_chunks_multiple_choice_indices():
    chunks = [
        _make_chunk(content="Hi", choice_index=0),
        _make_chunk(content="Hey", choice_index=1),
        _make_chunk(content=" there", choice_index=0),
        _make_chunk(content=" you", choice_index=1),
        _make_chunk(finish_reason="stop", choice_index=0),
        _make_chunk(finish_reason="stop", choice_index=1),
    ]
    result = aggregate_chat_stream_chunks(chunks)

    assert len(result["choices"]) == 2
    assert result["choices"][0]["index"] == 0
    assert result["choices"][0]["message"]["content"] == "Hi there"
    assert result["choices"][1]["index"] == 1
    assert result["choices"][1]["message"]["content"] == "Hey you"


@pytest.mark.asyncio
async def test_maybe_traced_gateway_call_basic(endpoint_config):
    traced_func = maybe_traced_gateway_call(mock_async_func, endpoint_config)
    result = await traced_func({"input": "test"})

    assert result == {"result": "success", "payload": {"input": "test"}}

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    # Find the gateway span
    span_name_to_span = {span.name: span for span in trace.data.spans}
    assert f"gateway/{endpoint_config.endpoint_name}" in span_name_to_span

    gateway_span = span_name_to_span[f"gateway/{endpoint_config.endpoint_name}"]
    assert gateway_span.attributes.get("endpoint_id") == "test-endpoint-id"
    assert gateway_span.attributes.get("endpoint_name") == "test-endpoint"
    # Input should be unwrapped (not nested under "payload" key)
    assert gateway_span.inputs == {"input": "test"}
    # No user metadata should be present in trace
    assert trace.info.request_metadata.get(TraceMetadataKey.AUTH_USERNAME) is None
    assert trace.info.request_metadata.get(TraceMetadataKey.AUTH_USER_ID) is None


@pytest.mark.asyncio
async def test_maybe_traced_gateway_call_with_user_metadata(endpoint_config):
    traced_func = maybe_traced_gateway_call(
        mock_async_func,
        endpoint_config,
        metadata={
            TraceMetadataKey.AUTH_USERNAME: "alice",
            TraceMetadataKey.AUTH_USER_ID: "123",
        },
    )
    result = await traced_func({"input": "test"})

    assert result == {"result": "success", "payload": {"input": "test"}}

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    span_name_to_span = {span.name: span for span in trace.data.spans}
    gateway_span = span_name_to_span[f"gateway/{endpoint_config.endpoint_name}"]

    assert gateway_span.attributes.get("endpoint_id") == "test-endpoint-id"
    assert gateway_span.attributes.get("endpoint_name") == "test-endpoint"
    # Input should be unwrapped (not nested under "payload" key)
    assert gateway_span.inputs == {"input": "test"}
    # User metadata should be in trace info, not span attributes
    assert trace.info.request_metadata.get(TraceMetadataKey.AUTH_USERNAME) == "alice"
    assert trace.info.request_metadata.get(TraceMetadataKey.AUTH_USER_ID) == "123"


@pytest.mark.asyncio
async def test_maybe_traced_gateway_call_without_experiment_id(endpoint_config_no_experiment):
    traced_func = maybe_traced_gateway_call(
        mock_async_func,
        endpoint_config_no_experiment,
        metadata={
            TraceMetadataKey.AUTH_USERNAME: "alice",
            TraceMetadataKey.AUTH_USER_ID: "123",
        },
    )

    # When experiment_id is None, maybe_traced_gateway_call returns the original function
    assert traced_func is mock_async_func

    result = await traced_func({"input": "test"})
    assert result == {"result": "success", "payload": {"input": "test"}}

    # No traces should be created
    traces = get_traces()
    assert len(traces) == 0


@pytest.mark.asyncio
async def test_maybe_traced_gateway_call_with_output_reducer(endpoint_config):
    async def mock_async_stream(payload):
        yield _make_chunk(content="Hello")
        yield _make_chunk(content=" world")
        yield _make_chunk(
            content=None,
            finish_reason="stop",
            usage=ChatUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )

    traced_func = maybe_traced_gateway_call(
        mock_async_stream,
        endpoint_config,
        output_reducer=aggregate_chat_stream_chunks,
    )

    # Consume the stream
    chunks = [
        chunk async for chunk in traced_func({"messages": [{"role": "user", "content": "hi"}]})
    ]
    assert len(chunks) == 3

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    span_name_to_span = {span.name: span for span in trace.data.spans}
    gateway_span = span_name_to_span[f"gateway/{endpoint_config.endpoint_name}"]

    # Input should be unwrapped (not nested under "payload" key)
    assert gateway_span.inputs == {"messages": [{"role": "user", "content": "hi"}]}

    # The output should be the reduced aggregated response, not raw chunks
    output = gateway_span.outputs
    assert output["object"] == "chat.completion"
    assert output["choices"][0]["message"]["content"] == "Hello world"
    assert output["choices"][0]["finish_reason"] == "stop"
    assert output["usage"]["total_tokens"] == 7


@pytest.mark.asyncio
async def test_maybe_traced_gateway_call_with_payload_kwarg(endpoint_config):
    async def mock_passthrough_func(action, payload, headers=None):
        return {"result": "success", "action": action, "payload": payload}

    traced_func = maybe_traced_gateway_call(mock_passthrough_func, endpoint_config)
    result = await traced_func(
        action="test_action", payload={"messages": [{"role": "user", "content": "hi"}]}, headers={}
    )

    assert result["result"] == "success"

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    span_name_to_span = {span.name: span for span in trace.data.spans}
    gateway_span = span_name_to_span[f"gateway/{endpoint_config.endpoint_name}"]

    # Input should be unwrapped to just the payload dict
    assert gateway_span.inputs == {"messages": [{"role": "user", "content": "hi"}]}
