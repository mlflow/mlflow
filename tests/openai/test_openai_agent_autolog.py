import copy
from contextlib import contextmanager
from typing import Optional
from unittest import mock

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice

import mlflow
from mlflow.entities import SpanType

from tests.openai.test_openai_autolog import client  # noqa: F401
from tests.tracing.helper import get_traces

try:
    from swarm import Agent, Swarm
except ImportError:
    pytest.skip("OpenAI Swarm not installed", allow_module_level=True)


@contextmanager
def mock_openai(oai_client, expected_responses):
    original = oai_client.chat.completions.create
    expected_responses = copy.deepcopy(expected_responses)

    def _mocked_create(*args, **kwargs):
        # We need to call the original SDK function because OpenAI autolog patches
        # it to generate a span for completion.
        original(*args, **kwargs)
        return expected_responses.pop(0)

    with mock.patch.object(oai_client.chat.completions, "create", side_effect=_mocked_create):
        yield


def test_autolog_swarm_agent(client):
    mlflow.openai.autolog()

    # NB: We have to mock the OpenAI SDK responses to make agent works
    DUMMY_RESPONSES = [
        _get_chat_completion(tool_call="transfer_to_spanish_agent"),
        _get_chat_completion(content="¡Hola! Estoy bien, gracias. ¿Y tú, cómo estás?"),
    ]

    swarm = Swarm(client=client)
    english_agent = Agent(name="English_Agent", instructions="You only speak English")
    spanish_agent = Agent(name="Spanish_Agent", instructions="You only speak Spanish")

    def transfer_to_spanish_agent():
        """Transfer spanish speaking users immediately"""
        return spanish_agent

    english_agent.functions.append(transfer_to_spanish_agent)

    with mock_openai(client, expected_responses=DUMMY_RESPONSES):
        messages = [{"role": "user", "content": "Hola.  ¿Como estás?"}]
        response = swarm.run(agent=english_agent, messages=messages)

    assert response.messages[-1]["content"] == "¡Hola! Estoy bien, gracias. ¿Y tú, cómo estás?"
    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "OK"
    spans = trace.data.spans
    assert len(spans) == 6  # 1 root + 1 function call + 2 get_chat_completion + 2 Completions
    assert spans[0].name == "run"
    assert spans[0].inputs["agent"]["name"] == "English_Agent"
    assert spans[0].inputs["messages"] == messages
    assert (
        spans[0].outputs["messages"][-1]["content"]
        == "¡Hola! Estoy bien, gracias. ¿Y tú, cómo estás?"
    )
    assert spans[1].name == "English_Agent.get_chat_completion"
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].name == "Completions_1"
    assert spans[2].parent_id == spans[1].span_id
    assert spans[3].name == "English_Agent.transfer_to_spanish_agent"
    assert spans[3].span_type == SpanType.TOOL
    assert spans[3].inputs == {}
    assert spans[3].outputs["name"] == "Spanish_Agent"
    assert spans[3].parent_id == spans[0].span_id
    assert spans[4].name == "Spanish_Agent.get_chat_completion"
    assert spans[4].parent_id == spans[0].span_id
    assert spans[5].name == "Completions_2"
    assert spans[5].parent_id == spans[4].span_id


def test_autolog_swarm_agent_with_context_variables(client):
    mlflow.openai.autolog()

    DUMMY_RESPONSES = [
        _get_chat_completion(tool_call="print_account_details"),
        _get_chat_completion(
            content="Hello, James! Your account details have been successfully printed."
        ),
    ]

    def instructions(context_variables):
        name = context_variables.get("name", "User")
        return f"You are a helpful agent. Greet the user by name ({name})."

    def print_account_details(context_variables):
        user_id = context_variables.get("user_id", None)
        name = context_variables.get("name", None)
        print(f"Account Details: {name} {user_id}")  # noqa: T201
        return "Success"

    swarm = Swarm(client=client)
    agent = Agent(name="Agent", instructions=instructions, functions=[print_account_details])

    with mock_openai(client, expected_responses=DUMMY_RESPONSES):
        context_variables = {"name": "James", "user_id": 123}
        messages = [{"role": "user", "content": "Print my account details!"}]

        response = swarm.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables,
        )

    assert (
        response.messages[-1]["content"]
        == "Hello, James! Your account details have been successfully printed."
    )
    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "OK"
    spans = trace.data.spans
    assert len(spans) == 6  # 1 root + 1 function call + 2 get_chat_completion + 2 Completions
    assert spans[0].name == "run"
    assert spans[0].inputs["agent"]["name"] == "Agent"
    assert spans[0].inputs["messages"] == messages
    assert (
        spans[0].outputs["messages"][-1]["content"]
        == "Hello, James! Your account details have been successfully printed."
    )
    assert spans[1].name == "Agent.get_chat_completion_1"
    assert spans[1].parent_id == spans[0].span_id
    assert spans[2].name == "Completions_1"
    assert spans[2].parent_id == spans[1].span_id
    assert spans[3].name == "Agent.print_account_details"
    assert spans[3].span_type == SpanType.TOOL
    assert spans[3].inputs["context_variables"] == context_variables
    assert spans[3].outputs == "Success"
    assert spans[3].parent_id == spans[0].span_id
    assert spans[4].name == "Agent.get_chat_completion_2"
    assert spans[4].parent_id == spans[0].span_id
    assert spans[5].name == "Completions_2"
    assert spans[5].parent_id == spans[4].span_id


def test_autolog_swarm_agent_tool_exception(client):
    mlflow.openai.autolog()

    DUMMY_RESPONSES = [_get_chat_completion(tool_call="always_fail")]

    swarm = Swarm(client=client)

    def always_fail():
        raise Exception("This function always fails")

    agent = Agent(name="Agent", instructions="You are an agent", functions=[always_fail])

    with mock_openai(client, expected_responses=DUMMY_RESPONSES):
        messages = [{"role": "user", "content": "Hi!"}]
        with pytest.raises(Exception, match="This function always fails"):
            swarm.run(agent=agent, messages=messages)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "ERROR"
    spans = trace.data.spans
    assert len(spans) == 4  # 1 root + 1 function call + 1 get_chat_completion + 1 Completions
    assert spans[3].span_type == SpanType.TOOL
    assert spans[3].status.status_code == "ERROR"
    assert spans[3].status.description == "Exception: This function always fails"
    assert spans[3].events[0].name == "exception"


def test_autolog_swarm_agent_completion_exception(client):
    mlflow.openai.autolog()

    swarm = Swarm(client=client)
    agent = Agent(name="Agent", instructions="You are an agent")

    with mock.patch.object(
        client.chat.completions, "create", side_effect=RuntimeError("Connection error")
    ):
        messages = [{"role": "user", "content": "Hi!"}]
        with pytest.raises(RuntimeError, match="Connection error"):
            swarm.run(agent=agent, messages=messages)

    traces = get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.info.status == "ERROR"
    spans = trace.data.spans
    assert len(spans) == 2  # 1 root + 1 get_chat_completion
    assert spans[1].status.status_code == "ERROR"
    assert spans[1].status.description == "RuntimeError: Connection error"
    assert spans[1].events[0].name == "exception"


def _get_chat_completion(content: Optional[str] = None, tool_call: Optional[str] = None):
    from openai.types.chat import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )

    if tool_call:
        choice = Choice(
            finish_reason="tool_calls",
            index=0,
            message=ChatCompletionMessage(
                content="",
                role="assistant",
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="123",
                        function=Function(
                            arguments="{}",
                            name=tool_call,
                        ),
                        type="function",
                    )
                ],
            ),
        )
    else:
        choice = Choice(
            finish_reason="stop",
            index=0,
            message=ChatCompletionMessage(content=content, role="assistant"),
        )

    return ChatCompletion(
        id="123", created=0, model="gpt-4o-mini", object="chat.completion", choices=[choice]
    )
