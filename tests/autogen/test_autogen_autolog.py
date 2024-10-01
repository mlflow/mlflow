import contextlib
import time
from typing import List
from unittest.mock import patch

import pytest
from autogen import ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent, io
from openai import APIConnectionError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice

import mlflow
from mlflow.entities.span import SpanType

from tests.helper_functions import start_mock_openai_server
from tests.tracing.helper import get_traces


@pytest.fixture(scope="module", autouse=True)
def mock_openai():
    with start_mock_openai_server() as base_url:
        yield base_url


@pytest.fixture
def llm_config(mock_openai):
    return {
        "config_list": [
            {
                "model": "gpt-4o-mini",
                "base_url": mock_openai,
                "api_key": "test",
                "temperature": 0.5,
                "max_tokens": 100,
            },
        ]
    }


@contextlib.contextmanager
def mock_user_input(messages: List[str]):
    with patch.object(io.IOStream.get_default(), "input", side_effect=messages):
        yield


def get_simple_agent(llm_config):
    assistant = ConversableAgent("agent", llm_config=llm_config)
    user_proxy = UserProxyAgent("user", code_execution_config=False)
    return assistant, user_proxy


def test_enable_disable_autolog(llm_config):
    mlflow.autogen.autolog()
    with mock_user_input(["Hi", "exit"]):
        assistant, user_proxy = get_simple_agent(llm_config)
        assistant.initiate_chat(user_proxy, message="foo")

    traces = get_traces()
    assert len(traces) == 1

    mlflow.autogen.autolog(disable=True)
    with mock_user_input(["Hi", "exit"]):
        assistant, user_proxy = get_simple_agent(llm_config)
        assistant.initiate_chat(user_proxy, message="foo")

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1


def test_tracing_agent(llm_config):
    mlflow.autogen.autolog()

    with mock_user_input(
        ["What is the capital of Tokyo?", "How long is it take from San Francisco?", "exit"]
    ):
        assistant, user_proxy = get_simple_agent(llm_config)
        response = assistant.initiate_chat(user_proxy, message="How can I help you today?")

    # Check if the initiate_chat method is patched
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].info.execution_time_ms > 0
    # 7 spans are expected:
    # initiate_chat
    #    |-- user_1
    #    |-- assistant_1 -- chat_completion
    #    |-- user_2
    #    |-- assistant_2 -- chat_completion
    assert len(traces[0].data.spans) == 7

    span_name_to_dict = {span.name: span for span in traces[0].data.spans}
    session_span = span_name_to_dict["initiate_chat"]
    assert session_span.name == "initiate_chat"
    assert session_span.span_type == SpanType.UNKNOWN
    assert session_span.inputs["message"] == "How can I help you today?"
    assert session_span.outputs["chat_history"] == response.chat_history
    user_span = span_name_to_dict["user_1"]
    assert user_span.span_type == SpanType.AGENT
    assert user_span.parent_id == session_span.span_id
    assert user_span.inputs["message"] == "How can I help you today?"
    assert user_span.outputs["message"]["content"] == "What is the capital of Tokyo?"
    agent_span = span_name_to_dict["agent_1"]
    assert agent_span.span_type == SpanType.AGENT
    assert agent_span.parent_id == session_span.span_id
    assert agent_span.inputs["message"]["content"] == "What is the capital of Tokyo?"
    assert agent_span.outputs is not None
    llm_span = span_name_to_dict["chat_completion_1"]
    assert llm_span.span_type == SpanType.LLM
    assert llm_span.parent_id == agent_span.span_id
    assert llm_span.inputs["messages"][-1]["content"] == "What is the capital of Tokyo?"
    assert llm_span.outputs is not None
    assert llm_span.attributes["cost"] >= 0
    user_span_2 = span_name_to_dict["user_2"]
    assert user_span_2.parent_id == session_span.span_id
    agent_span_2 = span_name_to_dict["agent_2"]
    assert agent_span_2.parent_id == session_span.span_id
    llm_span_2 = span_name_to_dict["chat_completion_2"]
    assert llm_span_2.parent_id == agent_span_2.span_id


def test_tracing_agent_with_error():
    mlflow.autogen.autolog()

    invalid_llm_config = {
        "config_list": [
            {
                "model": "gpt-4o-mini",
                "base_url": "invalid_url",
                "api_key": "invalid",
            }
        ]
    }
    assistant = ConversableAgent("agent", llm_config=invalid_llm_config)
    user_proxy = UserProxyAgent("user", code_execution_config=False)

    with mock_user_input(["What is the capital of Tokyo?", "exit"]):
        with pytest.raises(APIConnectionError, match="Connection error"):
            assistant.initiate_chat(user_proxy, message="How can I help you today?")

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    assert traces[0].info.execution_time_ms > 0
    assert traces[0].data.spans[0].status.status_code == "ERROR"
    assert traces[0].data.spans[0].status.description == "Connection error."


def test_tracing_agent_multiple_chat_sessions(llm_config):
    mlflow.autogen.autolog()

    with mock_user_input(["Hi", "exit", "Hello", "exit", "Hola", "exit"]):
        assistant, user_proxy = get_simple_agent(llm_config)
        assistant.initiate_chat(user_proxy, message="foo")
        assistant.initiate_chat(user_proxy, message="bar")
        assistant.initiate_chat(user_proxy, message="baz")

    # Traces should be created for each chat session
    traces = get_traces()
    assert len(traces) == 3


def test_tracing_agent_with_function_calling(llm_config):
    mlflow.autogen.autolog()

    # Define a simple tool and register it with the assistant agent
    def sum(a: int, b: int) -> int:
        time.sleep(1)
        return a + b

    assistant = ConversableAgent(
        name="assistant",
        system_message="You are a helpful AI assistant. "
        "You can help with simple calculations. "
        "Return 'TERMINATE' when the task is done.",
        llm_config=llm_config,
    )
    user_proxy = ConversableAgent(
        name="tool_agent",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None
        and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
    )
    assistant.register_for_llm(name="sum", description="A simple sum calculator")(sum)
    user_proxy.register_for_execution(name="sum")(sum)

    # Start a chat session. We mock OpenAI response to simulate function calling response.
    with patch(
        "autogen.oai.client.OpenAIClient.create",
        side_effect=[
            ChatCompletion(
                id="chat_1",
                created=0,
                object="chat.completion",
                model="gpt-4o-mini",
                choices=[
                    Choice(
                        index=1,
                        finish_reason="stop",
                        message=ChatCompletionMessage(
                            role="assistant",
                            tool_calls=[
                                {
                                    "id": "call_1",
                                    "function": {"arguments": '{"a": 1, "b": 1}', "name": "sum"},
                                    "type": "function",
                                },
                            ],
                        ),
                    ),
                ],
            ),
            ChatCompletion(
                id="chat_2",
                created=0,
                object="chat.completion",
                model="gpt-4o-mini",
                choices=[
                    Choice(
                        index=2,
                        finish_reason="stop",
                        message=ChatCompletionMessage(
                            role="assistant",
                            content="The result of the calculation is 2. \n\nTERMINATE",
                        ),
                    ),
                ],
            ),
        ],
    ):
        response = user_proxy.initiate_chat(assistant, message="What is 1 + 1?")

    assert response.summary.startswith("The result of the calculation is 2.")

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    span_name_to_dict = {span.name: span for span in traces[0].data.spans}
    assistant_span = span_name_to_dict["assistant_1"]
    assert assistant_span.span_type == SpanType.AGENT
    tool_agent_span = span_name_to_dict["tool_agent"]
    assert tool_agent_span.span_type == SpanType.AGENT
    tool_span = span_name_to_dict["sum"]
    assert tool_span.span_type == SpanType.TOOL
    assert tool_span.parent_id == tool_agent_span.span_id
    assert tool_span.inputs["a"] == 1
    assert tool_span.inputs["b"] == 1
    assert tool_span.outputs == "2"
    assert tool_span.end_time_ns - tool_span.start_time_ns >= 1e9  # 1 second


@pytest.fixture
def tokyo_timezone(monkeypatch):
    # Set the timezone to Tokyo
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()

    yield

    # Reset the timezone
    monkeypatch.delenv("TZ")
    time.tzset()


def test_tracing_llm_completion_duration_timezone(llm_config, tokyo_timezone):
    # Test if the duration calculation for LLM completion is robust to timezone changes.
    mlflow.autogen.autolog()

    with mock_user_input(
        ["What is the capital of Tokyo?", "How long is it take from San Francisco?", "exit"]
    ):
        assistant, user_proxy = get_simple_agent(llm_config)
        assistant.initiate_chat(user_proxy, message="How can I help you today?")

    # Check if the initiate_chat method is patched
    traces = get_traces()
    span_name_to_dict = {span.name: span for span in traces[0].data.spans}
    llm_span = span_name_to_dict["chat_completion_1"]

    # We mock OpenAI LLM call so it should not take too long e.g. > 10 seconds. If it does,
    # it most likely a bug such as incorrect timezone handling.
    assert 0 < llm_span.end_time_ns - llm_span.start_time_ns <= 10e9

    # Check if the start time is in reasonable range
    root_span = span_name_to_dict["initiate_chat"]
    assert 0 < llm_span.start_time_ns - root_span.start_time_ns <= 1e9


def test_tracing_composite_agent(llm_config):
    # Composite agent can call initiate_chat() or generate_reply() method of its sub-agents.
    # This test is to ensure that won't create a new trace for the sub-agent's method call.
    mlflow.autogen.autolog()

    agent_1 = ConversableAgent("agent_1", llm_config=llm_config)
    agent_2 = ConversableAgent("agent_2", llm_config=llm_config)
    group_chat = GroupChat(
        agents=[agent_1, agent_2],
        messages=[],
        max_round=3,
        speaker_selection_method="round_robin",
    )
    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )
    agent_1.initiate_chat(group_chat_manager, message="Hello")

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    spans = traces[0].data.spans
    # 1 for the root initiate_chat, 2 for the messages and 2 for the corresponding LLM calls.
    assert len(spans) == 5
    span_names = {span.name for span in spans}
    assert span_names == {
        "initiate_chat",
        "agent_1",
        "chat_completion_1",
        "agent_2",
        "chat_completion_2",
    }
