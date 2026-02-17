import pytest

import mlflow
from mlflow.entities import SpanType
from mlflow.gateway.schemas.chat import StreamResponsePayload
from mlflow.gateway.tracing_utils import (
    _get_provider_span_info,
    aggregate_chat_stream_chunks,
    maybe_traced_gateway_call,
)
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey
from mlflow.tracing.distributed import get_tracing_context_headers_for_http_request
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.types.chat import ChatChoiceDelta, ChatChunkChoice, ChatUsage, Function, ToolCallDelta


@pytest.fixture
def gateway_experiment_id():
    return mlflow.create_experiment("gateway-test-endpoint")


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


# ---------------------------------------------------------------------------
# Tests for distributed tracing helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_provider_span_info_reads_child_span(endpoint_config):
    async def func_with_child_span(payload):
        with mlflow.start_span("provider/openai/gpt-4", span_type=SpanType.LLM) as child:
            child.set_attributes(
                {
                    SpanAttributeKey.CHAT_USAGE: {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                    SpanAttributeKey.MODEL: "gpt-4",
                    SpanAttributeKey.MODEL_PROVIDER: "openai",
                }
            )
        return {"result": "ok"}

    traced = maybe_traced_gateway_call(func_with_child_span, endpoint_config)
    await traced({"input": "test"})

    traces = get_traces()
    assert len(traces) == 1
    gateway_trace_id = traces[0].info.trace_id

    # After the trace is exported, spans are removed from InMemoryTraceManager,
    # so we expect empty here. The actual reading happens inside the wrapper
    # while the trace is still in memory.
    assert _get_provider_span_info(gateway_trace_id) == []


# ---------------------------------------------------------------------------
# Integration tests for distributed tracing via traceparent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_traced_gateway_call_with_traceparent(gateway_experiment_id):
    ep_config = GatewayEndpointConfig(
        endpoint_id="test-endpoint-id",
        endpoint_name="test-endpoint",
        experiment_id=gateway_experiment_id,
        models=[],
    )

    async def func_with_usage(payload):
        with mlflow.start_span("provider/openai/gpt-4", span_type=SpanType.LLM) as child:
            child.set_attributes(
                {
                    SpanAttributeKey.CHAT_USAGE: {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                    SpanAttributeKey.MODEL: "gpt-4",
                    SpanAttributeKey.MODEL_PROVIDER: "openai",
                }
            )
        return {"result": "ok"}

    # Step 1: Agent creates span and generates traceparent headers
    with mlflow.start_span("agent-root") as agent_span:
        headers = get_tracing_context_headers_for_http_request()
        agent_trace_id = agent_span.trace_id
        agent_span_id = agent_span.span_id

    # Step 2: Gateway processes request (no active agent span, simulating separate server)
    traced = maybe_traced_gateway_call(func_with_usage, ep_config, request_headers=headers)
    result = await traced({"input": "test"})

    assert result == {"result": "ok"}

    # Gateway trace should exist in the gateway experiment
    gateway_traces = TracingClient().search_traces(locations=[gateway_experiment_id])
    assert len(gateway_traces) == 1
    gateway_trace_id = gateway_traces[0].info.trace_id

    # The gateway trace should be separate from the agent trace
    assert gateway_trace_id != agent_trace_id

    # Agent trace should contain two distributed spans (gateway + provider)
    mlflow.flush_trace_async_logging()
    agent_trace = mlflow.get_trace(agent_trace_id)
    assert agent_trace is not None

    spans_by_name = {s.name: s for s in agent_trace.data.spans}
    assert "agent-root" in spans_by_name
    assert f"gateway/{ep_config.endpoint_name}" in spans_by_name
    assert "provider/openai/gpt-4" in spans_by_name

    # Gateway span: child of agent root, has endpoint attrs + link
    gw_span = spans_by_name[f"gateway/{ep_config.endpoint_name}"]
    assert gw_span.parent_id == agent_span_id
    assert gw_span.attributes.get("endpoint_id") == ep_config.endpoint_id
    assert gw_span.attributes.get("endpoint_name") == ep_config.endpoint_name
    assert gw_span.attributes.get(SpanAttributeKey.LINKED_GATEWAY_TRACE_ID) == gateway_trace_id

    # Provider span: child of gateway span, has provider attrs
    provider_span = spans_by_name["provider/openai/gpt-4"]
    assert provider_span.parent_id == gw_span.span_id
    assert provider_span.attributes.get(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }
    assert provider_span.attributes.get(SpanAttributeKey.MODEL) == "gpt-4"
    assert provider_span.attributes.get(SpanAttributeKey.MODEL_PROVIDER) == "openai"

    # Neither span should have request/response payloads
    assert gw_span.inputs is None
    assert gw_span.outputs is None
    assert provider_span.inputs is None
    assert provider_span.outputs is None


@pytest.mark.asyncio
async def test_maybe_traced_gateway_call_streaming_with_traceparent(gateway_experiment_id):
    ep_config = GatewayEndpointConfig(
        endpoint_id="test-endpoint-id",
        endpoint_name="test-endpoint",
        experiment_id=gateway_experiment_id,
        models=[],
    )

    async def mock_stream_with_usage(payload):
        with mlflow.start_span("provider/openai/gpt-4", span_type=SpanType.LLM) as child:
            child.set_attributes(
                {
                    SpanAttributeKey.CHAT_USAGE: {
                        "input_tokens": 20,
                        "output_tokens": 10,
                        "total_tokens": 30,
                    },
                    SpanAttributeKey.MODEL: "gpt-4",
                    SpanAttributeKey.MODEL_PROVIDER: "openai",
                }
            )
        yield _make_chunk(content="Hello")
        yield _make_chunk(content=" world", finish_reason="stop")

    # Agent creates headers
    with mlflow.start_span("agent-root") as agent_span:
        headers = get_tracing_context_headers_for_http_request()
        agent_trace_id = agent_span.trace_id
        agent_span_id = agent_span.span_id

    # Gateway processes request (separate context)
    traced = maybe_traced_gateway_call(
        mock_stream_with_usage,
        ep_config,
        output_reducer=aggregate_chat_stream_chunks,
        request_headers=headers,
    )
    chunks = [chunk async for chunk in traced({"input": "test"})]

    assert len(chunks) == 2

    # Gateway trace should exist
    gateway_traces = TracingClient().search_traces(locations=[gateway_experiment_id])
    assert len(gateway_traces) == 1
    gateway_trace_id = gateway_traces[0].info.trace_id
    assert gateway_trace_id != agent_trace_id

    # Agent trace should contain two distributed spans (gateway + provider)
    mlflow.flush_trace_async_logging()
    agent_trace = mlflow.get_trace(agent_trace_id)
    assert agent_trace is not None

    spans_by_name = {s.name: s for s in agent_trace.data.spans}
    assert "agent-root" in spans_by_name
    assert f"gateway/{ep_config.endpoint_name}" in spans_by_name
    assert "provider/openai/gpt-4" in spans_by_name

    # Gateway span: child of agent root, has endpoint attrs + link
    gw_span = spans_by_name[f"gateway/{ep_config.endpoint_name}"]
    assert gw_span.parent_id == agent_span_id
    assert gw_span.attributes.get("endpoint_id") == ep_config.endpoint_id
    assert gw_span.attributes.get("endpoint_name") == ep_config.endpoint_name
    assert gw_span.attributes.get(SpanAttributeKey.LINKED_GATEWAY_TRACE_ID) == gateway_trace_id

    # Provider span: child of gateway span, has provider attrs
    provider_span = spans_by_name["provider/openai/gpt-4"]
    assert provider_span.parent_id == gw_span.span_id
    assert provider_span.attributes.get(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 20,
        "output_tokens": 10,
        "total_tokens": 30,
    }
    assert provider_span.attributes.get(SpanAttributeKey.MODEL) == "gpt-4"
    assert provider_span.attributes.get(SpanAttributeKey.MODEL_PROVIDER) == "openai"

    # Neither span should have request/response payloads
    assert gw_span.inputs is None
    assert gw_span.outputs is None
    assert provider_span.inputs is None
    assert provider_span.outputs is None


@pytest.mark.asyncio
async def test_maybe_traced_gateway_call_with_traceparent_multiple_providers(gateway_experiment_id):
    ep_config = GatewayEndpointConfig(
        endpoint_id="test-endpoint-id",
        endpoint_name="test-endpoint",
        experiment_id=gateway_experiment_id,
        models=[],
    )

    async def func_with_multiple_providers(payload):
        with mlflow.start_span("provider/openai/gpt-4", span_type=SpanType.LLM) as child:
            child.set_attributes(
                {
                    SpanAttributeKey.CHAT_USAGE: {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                    SpanAttributeKey.MODEL: "gpt-4",
                    SpanAttributeKey.MODEL_PROVIDER: "openai",
                }
            )
        with mlflow.start_span("provider/anthropic/claude-3", span_type=SpanType.LLM) as child:
            child.set_attributes(
                {
                    SpanAttributeKey.CHAT_USAGE: {
                        "input_tokens": 20,
                        "output_tokens": 10,
                        "total_tokens": 30,
                    },
                    SpanAttributeKey.MODEL: "claude-3",
                    SpanAttributeKey.MODEL_PROVIDER: "anthropic",
                }
            )
        return {"result": "ok"}

    with mlflow.start_span("agent-root") as agent_span:
        headers = get_tracing_context_headers_for_http_request()
        agent_trace_id = agent_span.trace_id

    traced = maybe_traced_gateway_call(
        func_with_multiple_providers, ep_config, request_headers=headers
    )
    await traced({"input": "test"})

    mlflow.flush_trace_async_logging()
    agent_trace = mlflow.get_trace(agent_trace_id)
    assert agent_trace is not None

    spans_by_name = {s.name: s for s in agent_trace.data.spans}
    gw_span = spans_by_name[f"gateway/{ep_config.endpoint_name}"]

    # Both provider spans should be children of the gateway span
    provider_openai = spans_by_name["provider/openai/gpt-4"]
    assert provider_openai.parent_id == gw_span.span_id
    assert provider_openai.attributes.get(SpanAttributeKey.MODEL) == "gpt-4"
    assert provider_openai.attributes.get(SpanAttributeKey.MODEL_PROVIDER) == "openai"
    assert provider_openai.attributes.get(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }

    provider_anthropic = spans_by_name["provider/anthropic/claude-3"]
    assert provider_anthropic.parent_id == gw_span.span_id
    assert provider_anthropic.attributes.get(SpanAttributeKey.MODEL) == "claude-3"
    assert provider_anthropic.attributes.get(SpanAttributeKey.MODEL_PROVIDER) == "anthropic"
    assert provider_anthropic.attributes.get(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 20,
        "output_tokens": 10,
        "total_tokens": 30,
    }
