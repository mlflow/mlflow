import asyncio
import copy
import gc
import json
from unittest import mock

import openai
import pytest

try:
    import agents  # noqa: F401
except ImportError:
    pytest.skip("OpenAI SDK is not installed. Skipping tests.", allow_module_level=True)

from agents import Agent, Runner, function_tool, set_default_openai_client, trace
from agents.tracing import set_trace_processors
from agents.tracing.processors import default_processor
from agents.tracing.setup import get_trace_provider
from openai.types.responses.function_tool import FunctionTool
from openai.types.responses.response import Response
from openai.types.responses.response_completed_event import ResponseCompletedEvent
from openai.types.responses.response_output_item import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
)
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

import mlflow
from mlflow.entities import SpanType
from mlflow.openai._agent_tracer import MlflowOpenAgentTracingProcessor

from tests.tracing.helper import get_traces, purge_traces


@pytest.fixture(autouse=True)
def restore_default_trace_processors():
    yield
    # Restore the default OpenAI agents tracer after each test
    set_trace_processors([default_processor()])


def set_dummy_client(expected_responses):
    expected_responses = copy.deepcopy(expected_responses)

    async def _mocked_create(*args, **kwargs):
        return expected_responses.pop(0)

    async_client = openai.AsyncOpenAI(api_key="test")
    async_client.responses = mock.MagicMock()
    async_client.responses.create = _mocked_create

    set_default_openai_client(async_client)


@pytest.mark.asyncio
async def test_autolog_agent():
    mlflow.openai.autolog()

    # NB: We have to mock the OpenAI SDK responses to make agent works
    DUMMY_RESPONSES = [
        Response(
            id="123",
            created_at=12345678.0,
            error=None,
            model="gpt-4o-mini",
            object="response",
            instructions="Handoff to the appropriate agent based on the language of the request.",
            output=[
                ResponseFunctionToolCall(
                    id="123",
                    arguments="{}",
                    call_id="123",
                    name="transfer_to_spanish_agent",
                    type="function_call",
                    status="completed",
                )
            ],
            tools=[
                FunctionTool(
                    name="transfer_to_spanish_agent",
                    parameters={"type": "object", "properties": {}, "required": []},
                    type="function",
                    description="Handoff to the Spanish_Agent agent to handle the request.",
                    strict=False,
                ),
            ],
            tool_choice="auto",
            temperature=1,
            parallel_tool_calls=True,
        ),
        Response(
            id="123",
            created_at=12345678.0,
            error=None,
            model="gpt-4o-mini",
            object="response",
            instructions="You only speak Spanish",
            output=[
                ResponseOutputMessage(
                    id="123",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text="¡Hola! Estoy bien, gracias. ¿Y tú, cómo estás?",
                            type="output_text",
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            tools=[],
            tool_choice="auto",
            temperature=1,
            parallel_tool_calls=True,
        ),
    ]

    set_dummy_client(DUMMY_RESPONSES)

    english_agent = Agent(name="English Agent", instructions="You only speak English")
    spanish_agent = Agent(name="Spanish Agent", instructions="You only speak Spanish")
    triage_agent = Agent(
        name="Triage Agent",
        instructions="Handoff to the appropriate agent based on the language of the request.",
        handoffs=[spanish_agent, english_agent],
    )

    messages = [{"role": "user", "content": "Hola.  ¿Como estás?"}]
    response = await Runner.run(starting_agent=triage_agent, input=messages)

    assert response.final_output == "¡Hola! Estoy bien, gracias. ¿Y tú, cómo estás?"
    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "OK"
    assert json.loads(trace.info.request_preview) == messages
    assert json.loads(trace.info.response_preview) == response.final_output
    spans = trace.data.spans
    assert len(spans) > 5


@pytest.mark.asyncio
async def test_autolog_agent_tool_exception():
    mlflow.openai.autolog()

    DUMMY_RESPONSES = [
        Response(
            id="123",
            created_at=12345678.0,
            error=None,
            model="gpt-4o-mini",
            object="response",
            instructions="You are an agent.",
            output=[
                ResponseFunctionToolCall(
                    id="123",
                    arguments="{}",
                    call_id="123",
                    name="always_fail",
                    type="function_call",
                    status="completed",
                )
            ],
            tools=[
                FunctionTool(
                    name="always_fail",
                    parameters={"type": "object", "properties": {}, "required": []},
                    type="function",
                    strict=False,
                ),
            ],
            tool_choice="auto",
            temperature=1,
            parallel_tool_calls=True,
        ),
    ]

    @function_tool(failure_error_function=None)  # Set error function None to avoid retry
    def always_fail():
        raise Exception("This function always fails")

    set_dummy_client(DUMMY_RESPONSES * 3)

    agent = Agent(name="Agent", instructions="You are an agent", tools=[always_fail])

    with pytest.raises(Exception, match="This function always fails"):
        await Runner.run(agent, [{"role": "user", "content": "Hi!"}])

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "ERROR"
    spans = trace.data.spans
    assert len(spans) > 3


@pytest.mark.asyncio
async def test_autolog_agent_llm_exception():
    mlflow.openai.autolog()

    agent = Agent(name="Agent", instructions="You are an agent")
    messages = [{"role": "user", "content": "Hi!"}]

    # Mock OpenAI SDK to raise an exception
    async_client = openai.AsyncOpenAI(api_key="test")
    async_client.responses = mock.MagicMock()
    async_client.responses.create.side_effect = RuntimeError("Connection error")
    set_default_openai_client(async_client)

    with pytest.raises(RuntimeError, match="Connection error"):
        await Runner.run(agent, messages)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "ERROR"
    spans = trace.data.spans
    assert len(spans) > 2


@pytest.mark.asyncio
async def test_autolog_agent_with_manual_trace():
    mlflow.openai.autolog()

    # NB: We have to mock the OpenAI SDK responses to make agent works
    DUMMY_RESPONSES = [
        Response(
            id="123",
            created_at=12345678.0,
            error=None,
            model="gpt-4o-mini",
            object="response",
            instructions="Tell funny jokes.",
            output=[
                ResponseOutputMessage(
                    id="123",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text="Nice joke",
                            type="output_text",
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            tools=[],
            tool_choice="auto",
            temperature=1,
            parallel_tool_calls=True,
        ),
    ]

    set_dummy_client(DUMMY_RESPONSES)

    agent = Agent(name="Joke agent", instructions="Tell funny jokes.")

    with mlflow.start_span("Parent span"):
        with trace("Joke workflow"):
            response = await Runner.run(agent, "Tell me a joke")

    assert response.final_output == "Nice joke"
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    spans = traces[0].data.spans
    assert len(spans) > 4


@pytest.mark.asyncio
async def test_disable_enable_autolog():
    # NB: We have to mock the OpenAI SDK responses to make agent works
    DUMMY_RESPONSES = [
        Response(
            id="123",
            created_at=12345678.0,
            error=None,
            model="gpt-4o-mini",
            object="response",
            instructions="You are daddy joke teller.",
            output=[
                ResponseOutputMessage(
                    id="123",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text="Why is Peter Pan always flying?",
                            type="output_text",
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            tools=[],
            tool_choice="auto",
            temperature=1,
            parallel_tool_calls=True,
        ),
    ]

    set_dummy_client(DUMMY_RESPONSES * 5)

    agent = Agent(name="Agent", instructions="You are daddy joke teller")
    messages = [{"role": "user", "content": "Tell me a joke"}]

    # Enable tracing
    mlflow.openai.autolog()

    await Runner.run(agent, messages)

    traces = get_traces()
    assert len(traces) == 1
    purge_traces()

    # Enabling autolog again should not cause duplicate traces
    mlflow.openai.autolog()
    mlflow.openai.autolog()

    await Runner.run(agent, messages)

    traces = get_traces()
    assert len(traces) == 1
    purge_traces()

    # Disable tracing
    mlflow.openai.autolog(disable=True)

    await Runner.run(agent, messages)

    assert get_traces() == []


def _make_streamed_response(text: str) -> Response:
    return Response(
        id="123",
        created_at=12345678.0,
        error=None,
        model="gpt-4o-mini",
        object="response",
        output=[
            ResponseOutputMessage(
                id="123",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text=text,
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        tools=[],
        tool_choice="auto",
        temperature=1,
        parallel_tool_calls=True,
    )


def _patch_stream_response(events):
    async def _stream(*args, **kwargs):
        for event in events:
            yield event

    return mock.patch(
        "agents.models.openai_responses.OpenAIResponsesModel.stream_response",
        side_effect=_stream,
    )


@pytest.mark.asyncio
async def test_autolog_agent_run_streamed():
    mlflow.openai.autolog()

    final_response = _make_streamed_response("Hello! Streaming response.")
    stream_events = [
        ResponseTextDeltaEvent(
            content_index=0,
            delta="Hello! ",
            item_id="123",
            logprobs=[],
            output_index=0,
            sequence_number=0,
            type="response.output_text.delta",
        ),
        ResponseTextDeltaEvent(
            content_index=0,
            delta="Streaming response.",
            item_id="123",
            logprobs=[],
            output_index=0,
            sequence_number=1,
            type="response.output_text.delta",
        ),
        ResponseCompletedEvent(
            type="response.completed",
            response=final_response,
            sequence_number=2,
        ),
    ]

    set_dummy_client([])
    agent = Agent(name="assistant", instructions="You are a helpful assistant.")

    with _patch_stream_response(stream_events):
        result = Runner.run_streamed(agent, "Hello")
        async for _ in result.stream_events():
            pass

    assert result.final_output == "Hello! Streaming response."

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "OK"
    assert json.loads(trace.info.request_preview) == "Hello"
    assert json.loads(trace.info.response_preview) == "Hello! Streaming response."

    spans = trace.data.spans
    assert spans[0].name == "AgentRunner.run_streamed"
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].inputs == "Hello"
    assert spans[0].outputs == "Hello! Streaming response."
    # All non-root spans should be transitively descended from the streamed root
    span_ids = {span.span_id for span in spans}
    for span in spans[1:]:
        assert span.parent_id in span_ids


@pytest.mark.asyncio
async def test_autolog_agent_run_streamed_exception():
    mlflow.openai.autolog()

    async def _raise(*args, **kwargs):
        raise RuntimeError("Streaming failed")
        yield  # pragma: no cover - needed to make this an async generator

    set_dummy_client([])
    agent = Agent(name="assistant", instructions="You are helpful.")

    async def _consume_stream():
        result = Runner.run_streamed(agent, "Hello")
        async for _ in result.stream_events():
            pass

    with mock.patch(
        "agents.models.openai_responses.OpenAIResponsesModel.stream_response",
        side_effect=_raise,
    ):
        with pytest.raises(RuntimeError, match="Streaming failed"):
            await _consume_stream()

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "ERROR"
    spans = trace.data.spans
    assert spans[0].name == "AgentRunner.run_streamed"
    assert spans[0].status.status_code == "ERROR"
    assert any(event.name == "exception" for event in spans[0].events)


@pytest.mark.asyncio
async def test_autolog_agent_run_streamed_discarded_result_finalizes_span():
    # Exercises the weakref.finalize fallback when the user never iterates
    # `stream_events()`.
    mlflow.openai.autolog()

    set_dummy_client([])
    agent = Agent(name="assistant", instructions="You are helpful.")

    final_response = _make_streamed_response("Discarded.")
    stream_events = [
        ResponseCompletedEvent(
            type="response.completed",
            response=final_response,
            sequence_number=0,
        ),
    ]

    # Run inside a nested helper so `result` does not stay alive in the test
    # frame's local variables, then yield to the event loop several times so
    # the cancelled background task gets a chance to finish and release its
    # internal references to `result` before triggering GC.
    def _start_and_discard():
        with _patch_stream_response(stream_events):
            result = Runner.run_streamed(agent, "Hello")
            # Cancel the background task so it does not keep `result` alive.
            # Use the public `cancel()` API to stay version-agnostic across
            # `openai-agents` releases (older versions exposed the task as
            # `_run_impl_task`, newer ones as `run_loop_task`).
            result.cancel()

    _start_and_discard()
    for _ in range(5):
        await asyncio.sleep(0)
    gc.collect()

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert spans[0].name == "AgentRunner.run_streamed"
    assert spans[0].end_time_ns is not None
    assert spans[0].status.status_code == "OK"


def test_autolog_disable_openai_agent_tracer():
    def _get_processors():
        return get_trace_provider()._multi_processor._processors

    # Verify default processor exists before autolog
    assert any(not isinstance(p, MlflowOpenAgentTracingProcessor) for p in _get_processors())

    # When disable_openai_agent_tracer=False, the default OpenAI tracer should be preserved
    mlflow.openai.autolog(disable_openai_agent_tracer=False)
    processors = _get_processors()
    assert len(processors) >= 2
    assert any(isinstance(p, MlflowOpenAgentTracingProcessor) for p in processors)
    assert any(not isinstance(p, MlflowOpenAgentTracingProcessor) for p in processors)

    # By default, autolog should clear the OpenAI agents tracer
    mlflow.openai.autolog()
    processors = _get_processors()
    assert len(processors) == 1
    assert isinstance(processors[0], MlflowOpenAgentTracingProcessor)


def test_generation_span_attributes_stored_under_span_attribute_keys():
    from agents.tracing import GenerationSpanData

    from mlflow.openai._agent_tracer import _parse_span_data
    from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

    usage = {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 30,
    }
    span_data = mock.MagicMock(spec=GenerationSpanData)
    span_data.type = "generation"
    span_data.input = [{"role": "user", "content": "hello"}]
    span_data.output = [{"role": "assistant", "content": "hi"}]
    span_data.model = "gpt-4o-mini"
    span_data.model_config = {}
    span_data.usage = usage

    _, _, attributes = _parse_span_data(span_data)

    assert attributes.get(SpanAttributeKey.MODEL) == "gpt-4o-mini"
    assert "model" not in attributes
    assert attributes.get(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 30,
    }
    assert "usage" not in attributes


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (
            {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
            {"input_tokens": 5, "output_tokens": 10, "total_tokens": 15},
        ),
        (
            {"input_tokens": 5, "output_tokens": 10},
            {"input_tokens": 5, "output_tokens": 10},
        ),
        # cached tokens are nested inside input_tokens_details
        (
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
            },
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cache_read_input_tokens": 4,
                "cache_creation_input_tokens": 2,
            },
        ),
        # partial cache details — only cached_tokens present
        (
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 3},
            },
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cache_read_input_tokens": 3,
            },
        ),
        (None, None),
        ({}, None),
    ],
)
def test_parse_generation_usage(usage, expected):
    from mlflow.openai._agent_tracer import _parse_generation_usage

    assert _parse_generation_usage(usage) == expected


def test_calculate_span_cost_uses_generation_span_model_attribute():
    from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
    from mlflow.tracing.utils import calculate_span_cost

    usage = {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 30,
    }
    attribute_store = {
        SpanAttributeKey.MODEL: "some-model",
        SpanAttributeKey.CHAT_USAGE: usage,
        SpanAttributeKey.MODEL_PROVIDER: None,
    }
    mock_span = mock.MagicMock()
    mock_span.get_attribute.side_effect = attribute_store.get

    expected_cost = {"input_cost": 0.01, "output_cost": 0.02, "total_cost": 0.03}
    with mock.patch(
        "mlflow.tracing.utils.calculate_cost_by_model_and_token_usage",
        return_value=expected_cost,
    ) as mock_cost:
        result = calculate_span_cost(mock_span)

    mock_cost.assert_called_once_with("some-model", usage, None)
    assert result == expected_cost
