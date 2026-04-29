import asyncio
import json
from typing import Any

import pytest

import mlflow
from mlflow.entities import SpanType
from mlflow.gateway.constants import MLFLOW_GATEWAY_CALLER_HEADER
from mlflow.gateway.schemas.chat import StreamResponsePayload
from mlflow.gateway.tracing_utils import (
    _extract_caller,
    _get_model_span_info,
    aggregate_anthropic_messages_stream_chunks,
    aggregate_chat_stream_chunks,
    aggregate_gemini_stream_generate_content_chunks,
    aggregate_openai_responses_stream_chunks,
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
    experiment_name = "gateway-test-endpoint"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)


def get_traces():
    return TracingClient().search_traces(locations=[_get_experiment_id()])


@pytest.fixture
def endpoint_config():
    return GatewayEndpointConfig(
        endpoint_id="test-endpoint-id",
        endpoint_name="test-endpoint",
        experiment_id=_get_experiment_id(),
        usage_tracking=True,
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
async def test_maybe_traced_gateway_call_without_usage_tracking(endpoint_config_no_experiment):
    traced_func = maybe_traced_gateway_call(
        mock_async_func,
        endpoint_config_no_experiment,
        metadata={
            TraceMetadataKey.AUTH_USERNAME: "alice",
            TraceMetadataKey.AUTH_USER_ID: "123",
        },
    )

    # When usage_tracking is False, maybe_traced_gateway_call returns the original function
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
async def test_maybe_traced_gateway_call_with_message_format(endpoint_config):
    async def mock_async_func(payload):
        return {"id": "msg_1", "type": "message", "role": "assistant", "content": []}

    traced_func = maybe_traced_gateway_call(
        mock_async_func,
        endpoint_config,
        message_format="anthropic",
    )
    await traced_func({"messages": [{"role": "user", "content": "hi"}]})

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]

    span_name_to_span = {span.name: span for span in trace.data.spans}
    gateway_span = span_name_to_span[f"gateway/{endpoint_config.endpoint_name}"]
    assert gateway_span.get_attribute("mlflow.message.format") == "anthropic"


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
async def test_get_model_span_info_reads_child_span(endpoint_config):
    async def func_with_child_span(payload):
        with mlflow.start_span("provider/openai/gpt-4", span_type=SpanType.LLM) as child:
            child.set_attributes({
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                SpanAttributeKey.MODEL: "gpt-4",
                SpanAttributeKey.MODEL_PROVIDER: "openai",
            })
        return {"result": "ok"}

    traced = maybe_traced_gateway_call(func_with_child_span, endpoint_config)
    await traced({"input": "test"})

    traces = get_traces()
    assert len(traces) == 1
    gateway_trace_id = traces[0].info.trace_id

    # After the trace is exported, spans are removed from InMemoryTraceManager,
    # so we expect empty here. The actual reading happens inside the wrapper
    # while the trace is still in memory.
    assert _get_model_span_info(gateway_trace_id) == []


# ---------------------------------------------------------------------------
# Integration tests for distributed tracing via traceparent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_traced_gateway_call_with_traceparent(gateway_experiment_id):
    ep_config = GatewayEndpointConfig(
        endpoint_id="test-endpoint-id",
        endpoint_name="test-endpoint",
        experiment_id=gateway_experiment_id,
        usage_tracking=True,
        models=[],
    )

    async def func_with_usage(payload):
        with mlflow.start_span("provider/openai/gpt-4", span_type=SpanType.LLM) as child:
            child.set_attributes({
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                SpanAttributeKey.MODEL: "gpt-4",
                SpanAttributeKey.MODEL_PROVIDER: "openai",
            })
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

    # Flush to ensure all spans are written (batch processor may be active)
    mlflow.flush_trace_async_logging(terminate=True)

    # Gateway trace should exist in the gateway experiment
    gateway_traces = TracingClient().search_traces(locations=[gateway_experiment_id])
    assert len(gateway_traces) == 1
    gateway_trace_id = gateway_traces[0].info.trace_id

    # The gateway trace should be separate from the agent trace
    assert gateway_trace_id != agent_trace_id

    # Agent trace should contain two distributed spans (gateway + provider)
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

    # Provider span should preserve timing from the gateway trace
    gateway_provider_span = next(
        s for s in gateway_traces[0].data.spans if s.name == "provider/openai/gpt-4"
    )
    assert provider_span.start_time_ns == gateway_provider_span.start_time_ns
    assert provider_span.end_time_ns == gateway_provider_span.end_time_ns

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
        usage_tracking=True,
        models=[],
    )

    async def mock_stream_with_usage(payload):
        with mlflow.start_span("provider/openai/gpt-4", span_type=SpanType.LLM) as child:
            child.set_attributes({
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "total_tokens": 30,
                },
                SpanAttributeKey.MODEL: "gpt-4",
                SpanAttributeKey.MODEL_PROVIDER: "openai",
            })
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

    # Flush to ensure all spans are written (batch processor may be active)
    mlflow.flush_trace_async_logging(terminate=True)

    # Gateway trace should exist
    gateway_traces = TracingClient().search_traces(locations=[gateway_experiment_id])
    assert len(gateway_traces) == 1
    gateway_trace_id = gateway_traces[0].info.trace_id
    assert gateway_trace_id != agent_trace_id

    # Agent trace should contain two distributed spans (gateway + provider)
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

    # Provider span should preserve timing from the gateway trace
    gateway_provider_span = next(
        s for s in gateway_traces[0].data.spans if s.name == "provider/openai/gpt-4"
    )
    assert provider_span.start_time_ns == gateway_provider_span.start_time_ns
    assert provider_span.end_time_ns == gateway_provider_span.end_time_ns

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
        usage_tracking=True,
        models=[],
    )

    async def func_with_multiple_providers(payload):
        with mlflow.start_span("provider/openai/gpt-4", span_type=SpanType.LLM) as child:
            child.set_attributes({
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                SpanAttributeKey.MODEL: "gpt-4",
                SpanAttributeKey.MODEL_PROVIDER: "openai",
            })
        with mlflow.start_span("provider/anthropic/claude-3", span_type=SpanType.LLM) as child:
            child.set_attributes({
                SpanAttributeKey.CHAT_USAGE: {
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "total_tokens": 30,
                },
                SpanAttributeKey.MODEL: "claude-3",
                SpanAttributeKey.MODEL_PROVIDER: "anthropic",
            })
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

    # Provider spans should preserve timing from the gateway trace
    gateway_traces = TracingClient().search_traces(locations=[gateway_experiment_id])
    assert len(gateway_traces) == 1
    gw_spans_by_name = {s.name: s for s in gateway_traces[0].data.spans}

    gw_openai = gw_spans_by_name["provider/openai/gpt-4"]
    assert provider_openai.start_time_ns == gw_openai.start_time_ns
    assert provider_openai.end_time_ns == gw_openai.end_time_ns

    gw_anthropic = gw_spans_by_name["provider/anthropic/claude-3"]
    assert provider_anthropic.start_time_ns == gw_anthropic.start_time_ns
    assert provider_anthropic.end_time_ns == gw_anthropic.end_time_ns


# ---------------------------------------------------------------------------
# Tests for aggregate_anthropic_messages_stream_chunks
# ---------------------------------------------------------------------------


def _sse(event: dict[str, Any]) -> bytes:
    """Encode a single event dict as an SSE data line."""
    return f"data: {json.dumps(event)}\n".encode()


def _msg_start(msg_id: str, model: str, input_tokens: int | None = None) -> bytes:
    usage = {"input_tokens": input_tokens} if input_tokens is not None else {}
    return _sse({
        "type": "message_start",
        "message": {"id": msg_id, "model": model, "role": "assistant", "usage": usage},
    })


def _text_block_start(index: int) -> bytes:
    return _sse({
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    })


def _text_delta(index: int, text: str) -> bytes:
    return _sse({
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    })


def _tool_block_start(index: int, tool_id: str, name: str) -> bytes:
    return _sse({
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
    })


def _tool_delta(index: int, partial_json: str) -> bytes:
    return _sse({
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    })


def _msg_delta(stop_reason: str, output_tokens: int, stop_sequence: str | None = None) -> bytes:
    return _sse({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
        "usage": {"output_tokens": output_tokens},
    })


def test_aggregate_anthropic_messages_stream_chunks_empty():
    assert aggregate_anthropic_messages_stream_chunks([]) is None


def test_aggregate_anthropic_messages_stream_chunks_no_parseable_events():
    chunks = [b"event: ping\n", b"data: [DONE]\n"]
    assert aggregate_anthropic_messages_stream_chunks(chunks) is None


def test_aggregate_anthropic_messages_stream_chunks_text():
    chunks = [
        _msg_start("msg_1", "claude-3-5-sonnet-20241022", input_tokens=10),
        _text_block_start(0),
        _text_delta(0, "Hello"),
        _text_delta(0, " world"),
        _sse({"type": "content_block_stop", "index": 0}),
        _msg_delta("end_turn", output_tokens=5, stop_sequence=None),
        _sse({"type": "message_stop"}),
    ]
    result = aggregate_anthropic_messages_stream_chunks(chunks)

    assert result["id"] == "msg_1"
    assert result["type"] == "message"
    assert result["role"] == "assistant"
    assert result["model"] == "claude-3-5-sonnet-20241022"
    assert result["stop_reason"] == "end_turn"
    assert result["stop_sequence"] is None
    assert result["content"] == [{"type": "text", "text": "Hello world"}]
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_aggregate_anthropic_messages_stream_chunks_tool_use():
    chunks = [
        _msg_start("msg_2", "claude-3-5-sonnet-20241022", input_tokens=20),
        _tool_block_start(0, "toolu_abc", "get_weather"),
        _tool_delta(0, '{"city"'),
        _tool_delta(0, ': "Paris"}'),
        _sse({"type": "content_block_stop", "index": 0}),
        _msg_delta("tool_use", output_tokens=15, stop_sequence=None),
    ]
    result = aggregate_anthropic_messages_stream_chunks(chunks)

    assert result["stop_reason"] == "tool_use"
    assert len(result["content"]) == 1
    block = result["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "toolu_abc"
    assert block["name"] == "get_weather"
    assert block["input"] == {"city": "Paris"}
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 15}


def test_aggregate_anthropic_messages_stream_chunks_mixed_content():
    chunks = [
        _msg_start("msg_3", "claude-3-5-sonnet-20241022", input_tokens=30),
        _text_block_start(0),
        _text_delta(0, "Let me check that."),
        _sse({"type": "content_block_stop", "index": 0}),
        _tool_block_start(1, "toolu_xyz", "search"),
        _tool_delta(1, '{"q": "mlflow"}'),
        _sse({"type": "content_block_stop", "index": 1}),
        _msg_delta("tool_use", output_tokens=25, stop_sequence=None),
    ]
    result = aggregate_anthropic_messages_stream_chunks(chunks)

    assert len(result["content"]) == 2
    assert result["content"][0] == {"type": "text", "text": "Let me check that."}
    assert result["content"][1] == {
        "type": "tool_use",
        "id": "toolu_xyz",
        "name": "search",
        "input": {"q": "mlflow"},
    }


def test_aggregate_anthropic_messages_stream_chunks_multiple_chunks_per_sse():
    # Multiple SSE events packed into one bytes chunk (newline-separated)
    combined = (
        _msg_start("msg_4", "claude-3-5-sonnet-20241022", input_tokens=5)
        + _text_block_start(0)
        + _text_delta(0, "Hi")
        + _msg_delta("end_turn", output_tokens=2, stop_sequence=None)
    )
    result = aggregate_anthropic_messages_stream_chunks([combined])

    assert result["id"] == "msg_4"
    assert result["content"] == [{"type": "text", "text": "Hi"}]
    assert result["usage"] == {"input_tokens": 5, "output_tokens": 2}


@pytest.mark.parametrize(
    ("raw_json", "expected_input"),
    [
        ('{"key": "val"}', {"key": "val"}),
        ("", {}),
        ("not-valid-json", {}),
    ],
)
def test_aggregate_anthropic_messages_stream_chunks_tool_input_edge_cases(raw_json, expected_input):
    chunks = [
        _msg_start("msg_5", "claude-3-5-sonnet-20241022"),
        _tool_block_start(0, "t1", "fn"),
    ]
    if raw_json:
        chunks.append(_tool_delta(0, raw_json))
    chunks.append(_msg_delta("tool_use", output_tokens=1))

    result = aggregate_anthropic_messages_stream_chunks(chunks)
    assert result["content"][0]["input"] == expected_input


def test_aggregate_anthropic_messages_stream_chunks_split_sse_lines():
    # Simulate an aiohttp byte chunk that splits a "data:" SSE line mid-way.
    # All events should still be parsed correctly after concatenation.
    msg_start_bytes = _msg_start("msg_split", "claude-3-5-sonnet-20241022", input_tokens=3)
    mid = len(msg_start_bytes) // 2
    chunks = [
        msg_start_bytes[:mid],
        msg_start_bytes[mid:] + _msg_delta("end_turn", output_tokens=1),
    ]
    result = aggregate_anthropic_messages_stream_chunks(chunks)

    assert result is not None
    assert result["id"] == "msg_split"
    assert result["usage"] == {"input_tokens": 3, "output_tokens": 1}


def test_aggregate_anthropic_messages_stream_chunks_cache_tokens():
    chunks = [
        _sse({
            "type": "message_start",
            "message": {
                "id": "msg_cache",
                "model": "claude-3-5-sonnet-20241022",
                "role": "assistant",
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 2,
                },
            },
        }),
        _msg_delta("end_turn", output_tokens=8),
    ]
    result = aggregate_anthropic_messages_stream_chunks(chunks)

    assert result["usage"] == {
        "input_tokens": 10,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 2,
        "output_tokens": 8,
    }


# ---------------------------------------------------------------------------
# Tests for aggregate_openai_responses_stream_chunks
# ---------------------------------------------------------------------------

_RESPONSES_CREATED = (
    b'data: {"type":"response.created","response":{"id":"resp_1","object":"response",'
    b'"created_at":1741290958,"status":"in_progress","output":[],"usage":null}}\n'
)
_RESPONSES_TEXT_DELTA = (
    b'data: {"type":"response.output_text.delta","item_id":"msg_1",'
    b'"output_index":0,"content_index":0,"delta":"Hi"}\n'
)
_RESPONSES_TEXT_DONE = (
    b'data: {"type":"response.output_text.done","item_id":"msg_1",'
    b'"output_index":0,"content_index":0,"text":"Hi there!"}\n'
)
_RESPONSES_COMPLETED = (
    b'data: {"type":"response.completed","response":{"id":"resp_1","object":"response",'
    b'"created_at":1741290958,"status":"completed",'
    b'"output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant",'
    b'"content":[{"type":"output_text","text":"Hi there!","annotations":[]}]}],'
    b'"usage":{"input_tokens":37,"output_tokens":11,"total_tokens":48}}}\n'
)


def test_aggregate_openai_responses_stream_chunks_empty():
    assert aggregate_openai_responses_stream_chunks([]) is None


def test_aggregate_openai_responses_stream_chunks_no_completed_event():
    chunks = [_RESPONSES_CREATED, _RESPONSES_TEXT_DELTA]
    assert aggregate_openai_responses_stream_chunks(chunks) is None


def test_aggregate_openai_responses_stream_chunks_basic():
    chunks = [
        _RESPONSES_CREATED,
        _RESPONSES_TEXT_DELTA,
        _RESPONSES_TEXT_DONE,
        _RESPONSES_COMPLETED,
    ]
    result = aggregate_openai_responses_stream_chunks(chunks)

    assert result["id"] == "resp_1"
    assert result["object"] == "response"
    assert result["status"] == "completed"
    assert len(result["output"]) == 1
    assert result["output"][0]["role"] == "assistant"
    assert result["output"][0]["content"][0]["text"] == "Hi there!"
    assert result["usage"] == {"input_tokens": 37, "output_tokens": 11, "total_tokens": 48}


def test_aggregate_openai_responses_stream_chunks_split_sse_lines():
    # Simulate aiohttp yielding a chunk that splits the data: line mid-way.
    mid = len(_RESPONSES_COMPLETED) // 2
    chunks = [
        _RESPONSES_CREATED,
        _RESPONSES_COMPLETED[:mid],
        _RESPONSES_COMPLETED[mid:],
    ]
    result = aggregate_openai_responses_stream_chunks(chunks)

    assert result is not None
    assert result["id"] == "resp_1"
    assert result["status"] == "completed"


def test_aggregate_openai_responses_stream_chunks_returns_completed_response():
    # When multiple events are packed into a single bytes chunk, the
    # completed response is still extracted correctly.
    combined = _RESPONSES_CREATED + _RESPONSES_TEXT_DELTA + _RESPONSES_COMPLETED
    result = aggregate_openai_responses_stream_chunks([combined])

    assert result["status"] == "completed"
    assert result["usage"]["total_tokens"] == 48


# ---------------------------------------------------------------------------
# Tests for aggregate_gemini_stream_generate_content_chunks
# ---------------------------------------------------------------------------


def _gemini_sse(event: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(event)}\n".encode()


def _gemini_text_chunk(text: str, finish_reason: str | None = None) -> bytes:
    candidate: dict[str, Any] = {"content": {"parts": [{"text": text}], "role": "model"}}
    if finish_reason:
        candidate["finishReason"] = finish_reason
    return _gemini_sse({"candidates": [candidate]})


def test_aggregate_gemini_stream_chunks_empty():
    assert aggregate_gemini_stream_generate_content_chunks([]) is None


def test_aggregate_gemini_stream_chunks_no_parseable_events():
    chunks = [b"event: ping\n", b"data: [DONE]\n"]
    assert aggregate_gemini_stream_generate_content_chunks(chunks) is None


def test_aggregate_gemini_stream_chunks_text():
    chunks = [
        _gemini_text_chunk("Hello"),
        _gemini_text_chunk(" world", finish_reason="STOP"),
        _gemini_sse({
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            }
        }),
    ]
    result = aggregate_gemini_stream_generate_content_chunks(chunks)

    assert len(result["candidates"]) == 1
    cand = result["candidates"][0]
    assert cand["content"]["parts"] == [{"text": "Hello world"}]
    assert cand["content"]["role"] == "model"
    assert cand["finishReason"] == "STOP"
    assert result["usageMetadata"] == {
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
        "totalTokenCount": 15,
    }


def test_aggregate_gemini_stream_chunks_tool_call():
    chunks = [
        _gemini_sse({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ]
        }),
        _gemini_sse({
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 12,
                "totalTokenCount": 20,
            }
        }),
    ]
    result = aggregate_gemini_stream_generate_content_chunks(chunks)

    cand = result["candidates"][0]
    assert cand["content"]["parts"] == [
        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
    ]
    assert cand["finishReason"] == "STOP"


def test_aggregate_gemini_stream_chunks_split_sse_lines():
    chunk_bytes = _gemini_text_chunk("Hi", finish_reason="STOP")
    mid = len(chunk_bytes) // 2
    result = aggregate_gemini_stream_generate_content_chunks([chunk_bytes[:mid], chunk_bytes[mid:]])

    assert result is not None
    assert result["candidates"][0]["content"]["parts"] == [{"text": "Hi"}]


@pytest.mark.parametrize(
    ("finish_reasons", "expected"),
    [
        ([None, None, "STOP"], "STOP"),
        ([None, "stop", None], "stop"),
        ([None, None, None], None),
    ],
)
def test_aggregate_gemini_stream_chunks_finish_reason(finish_reasons, expected):
    chunks = [_gemini_text_chunk(f"t{i}", finish_reason=fr) for i, fr in enumerate(finish_reasons)]
    result = aggregate_gemini_stream_generate_content_chunks(chunks)
    assert result["candidates"][0]["finishReason"] == expected


# ---------------------------------------------------------------------------
# _extract_caller tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (None, None),
        ({}, None),
        ({"User-Agent": "openai-python/1.0.0"}, "openai-python"),
        ({"user-agent": "claude-cli/1.2.3"}, "claude-cli"),
        ({"User-Agent": "GeminiCLI/1.0 (Linux; x64)"}, "GeminiCLI"),
        ({"User-Agent": "anthropic/0.20.0 CPython/3.11.0 Darwin/23.0.0"}, "anthropic"),
        (
            {MLFLOW_GATEWAY_CALLER_HEADER: "judge", "User-Agent": "openai-python/1.0.0"},
            "judge",
        ),
        ({MLFLOW_GATEWAY_CALLER_HEADER: "judge"}, "judge"),
        ({"User-Agent": "   "}, None),
    ],
)
def test_extract_caller(headers, expected):
    assert _extract_caller(headers) == expected


def test_maybe_traced_gateway_call_records_caller(endpoint_config):
    async def fake_func(payload):
        return {"ok": True}

    traced = maybe_traced_gateway_call(
        fake_func,
        endpoint_config,
        request_headers={"User-Agent": "openai-python/1.0.0"},
    )

    asyncio.get_event_loop().run_until_complete(traced({"prompt": "hi"}))

    traces = get_traces()
    assert traces
    assert traces[0].info.request_metadata.get(TraceMetadataKey.GATEWAY_CALLER) == "openai-python"


def test_maybe_traced_gateway_call_no_caller_when_no_headers(endpoint_config):
    async def fake_func(payload):
        return {"ok": True}

    traced = maybe_traced_gateway_call(fake_func, endpoint_config, request_headers=None)

    asyncio.get_event_loop().run_until_complete(traced({"prompt": "hi"}))

    traces = get_traces()
    assert traces
    assert TraceMetadataKey.GATEWAY_CALLER not in (traces[0].info.request_metadata or {})
