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
from openai.types.responses.function_tool import FunctionTool
from openai.types.responses.response import Response
from openai.types.responses.response_output_item import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
)
from openai.types.responses.response_output_text import ResponseOutputText

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces, purge_traces


def set_dummy_client(expected_responses):
    expected_responses = copy.deepcopy(expected_responses)

    async def _mocked_create(*args, **kwargs):
        return expected_responses.pop(0)

    async_client = openai.AsyncOpenAI(api_key="test")
    async_client.responses = mock.MagicMock()
    async_client.responses.create = _mocked_create

    set_default_openai_client(async_client)


@pytest.fixture(autouse=True)
def disable_default_tracing():
    # Disable default OpenAI tracer
    set_trace_processors([])


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
    assert len(spans) == 6  # 1 root + 2 agent + 1 handoff + 2 response
    assert spans[0].name == "AgentRunner.run"
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].inputs == messages
    assert spans[0].outputs == response.final_output
    assert spans[1].name == "Triage Agent"
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].name == "Response_1"
    assert spans[2].parent_id == spans[1].span_id
    assert spans[2].inputs == [{"role": "user", "content": "Hola.  ¿Como estás?"}]
    assert spans[2].outputs == [
        {
            "id": "123",
            "arguments": "{}",
            "call_id": "123",
            "name": "transfer_to_spanish_agent",
            "type": "function_call",
            "status": "completed",
        }
    ]
    assert spans[2].attributes["temperature"] == 1
    assert spans[3].name == "Handoff"
    assert spans[3].span_type == SpanType.CHAIN
    assert spans[3].parent_id == spans[1].span_id
    assert spans[4].name == "Spanish Agent"
    assert spans[4].parent_id == spans[0].span_id
    assert spans[5].name == "Response_2"
    assert spans[5].parent_id == spans[4].span_id

    # Validate chat attributes
    assert spans[2].attributes[SpanAttributeKey.CHAT_TOOLS] == [
        {
            "function": {
                "description": "Handoff to the Spanish_Agent agent to handle the request.",
                "name": "transfer_to_spanish_agent",
                "parameters": {
                    "additionalProperties": None,
                    "properties": {},
                    "required": [],
                    "type": "object",
                },
                "strict": False,
            },
            "type": "function",
        },
    ]
    assert SpanAttributeKey.CHAT_TOOLS not in spans[5].attributes


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
    assert len(spans) == 4  # 1 root + 1 function call + 1 get_chat_completion + 1 Completions
    assert spans[3].span_type == SpanType.TOOL
    assert spans[3].status.status_code == "ERROR"
    assert spans[3].status.description == "Error running tool"
    assert spans[3].events[0].name == "exception"


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
    assert len(spans) == 3
    assert spans[0].name == "AgentRunner.run"
    assert spans[2].status.status_code == "ERROR"
    assert spans[2].status.description == "Error getting response"
    assert spans[2].events[0].name == "exception"
    assert spans[2].events[0].attributes["exception.message"] == "Error getting response"
    assert spans[2].events[0].attributes["exception.stacktrace"] == '{"error": "Connection error"}'


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
    assert len(spans) == 5
    assert spans[0].name == "Parent span"
    assert spans[1].name == "Joke workflow"
    assert spans[2].name == "AgentRunner.run"
    assert spans[3].name == "Joke agent"
    assert spans[4].name == "Response"


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
