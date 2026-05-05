import copy
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
from openai.types.responses.response_output_item import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
)
from openai.types.responses.response_output_text import ResponseOutputText

import mlflow
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
