import json

from openai.types.chat.chat_completion import ChatCompletion, Choice
import pytest
from unittest import mock

import mlflow
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.tracing.constant import TraceMetadataKey

from tests.openai.conftest import is_v1
from tests.openai.test_openai_autolog import client
from tests.tracing.helper import get_traces


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_autolog_swarm_agent(client):
    from swarm import Swarm, Agent
    from swarm.types import ChatCompletionMessage, ChatCompletionMessageToolCall, Function

    mlflow.openai.autolog()

    _DUMMY_RESPONSES = [
        ChatCompletion(
            id="123",
            created=0,
            model="gpt-4o-mini",
            object="chat.completion",
            choices=[
                Choice(
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
                                    name="transfer_to_spanish_agent"
                                ),
                                type="function"
                            )
                        ],
                    )
                )
            ]
        ),
        ChatCompletion(
            id="456",
            created=0,
            model="gpt-4o-mini",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="¡Hola! Estoy bien, gracias. ¿Y tú, cómo estás?",
                        role="assistant",
                        tool_calls=None,
                    )
                )
            ]
        ),
    ]

    # NB: We have to mock the exact responses to make agent works
    original = client.chat.completions.create
    with mock.patch.object(client.chat.completions, "create") as mock_create:
        def _mocked_create(*args, **kwargs):
            # We also need to run the original call because OpenAI autolog patches the original
            # function to generate a span for completion. However, we need to tweak the response
            # to make the agent work.
            original(*args, **kwargs)
            response = _DUMMY_RESPONSES.pop(0)
            return response

        mock_create.side_effect = _mocked_create

        swarm = Swarm(client=client)

        english_agent = Agent(name="English_Agent", instructions="You only speak English")
        spanish_agent = Agent(name="Spanish_Agent", instructions="You only speak Spanish")

        def transfer_to_spanish_agent():
            """Transfer spanish speaking users immediately"""
            return spanish_agent

        english_agent.functions.append(transfer_to_spanish_agent)

        messages = [{"role": "user", "content": "Hola.  ¿Como estás?"}]
        response = swarm.run(agent=english_agent, messages=messages)

        traces = get_traces()
        assert len(traces) == 1
        trace = traces[0]
        assert trace.info.status == "OK"
        spans = trace.data.spans
        assert len(spans) == 6  # 1 root + 1 function call + 2 get_chat_completion + 2 Completions
        assert spans[0].name == "run"
        assert spans[0].inputs["agent"]["name"] == "English_Agent"
        assert spans[0].inputs["messages"] == messages
        assert spans[0].outputs["messages"][-1]["content"] == "¡Hola! Estoy bien, gracias. ¿Y tú, cómo estás?"
        assert spans[1].name == "English_Agent.get_chat_completion"
        assert spans[1].parent_id == spans[0].span_id
        assert spans[2].name == "Completions_1"
        assert spans[2].parent_id == spans[1].span_id
        assert spans[3].name == "English_Agent.transfer_to_spanish_agent"
        assert spans[3].span_type == SpanType.TOOL
        assert spans[3].parent_id == spans[0].span_id
        assert spans[4].name == "Spanish_Agent.get_chat_completion"
        assert spans[4].parent_id == spans[0].span_id
        assert spans[5].name == "Completions_2"
        assert spans[5].parent_id == spans[4].span_id


@pytest.mark.skipif(not is_v1, reason="Requires OpenAI SDK v1")
def test_autolog_swarm_agent_with_context_variables():
    pass


def test_autolog_swarm_like_agent_diy():
    pass