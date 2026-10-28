from typing import Any

import pytest
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

import mlflow
from mlflow.entities import SpanType
from mlflow.genai.utils.trace_utils import (
    extract_request_from_trace,
    extract_response_from_trace,
    extract_tools_called_from_trace,
)

from tests.tracing.helper import get_traces


class ScriptedModel(BaseLlm):
    replies: list[Any]

    async def generate_content_async(self, llm_request, stream=False):
        yield LlmResponse(content=types.Content(role="model", parts=self.replies.pop(0)))


def _call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def _text(text, thought=None):
    return types.Part(text=text, thought=thought)


def log_food(food: str, qty: int) -> dict[str, Any]:
    """Log food to the diary."""
    return {"status": "success", "kcal": 78 * qty}


def get_remaining() -> dict[str, Any]:
    """Remaining calories today."""
    return {"remaining_kcal": 1344}


def _agent(replies, tools=()):
    return LlmAgent(
        name="nutrition",
        model=ScriptedModel(model="scripted", replies=replies),
        instruction="nutrition",
        tools=list(tools),
    )


async def _send(runner, session_id, part):
    message = types.Content(role="user", parts=[part])
    return [
        event
        async for event in runner.run_async(user_id="u", session_id=session_id, new_message=message)
    ]


async def _session(runner):
    return (await runner.session_service.create_session(app_name=runner.app_name, user_id="u")).id


def _root(trace):
    return next(span for span in trace.data.spans if span.parent_id is None)


@pytest.mark.asyncio
async def test_root_span_records_user_message_and_reply():
    mlflow.google_adk.autolog()
    runner = InMemoryRunner(
        agent=_agent([[_call("get_remaining")], [_text("1,344 kcal left.")]], tools=[get_remaining])
    )
    session_id = await _session(runner)

    await _send(runner, session_id, _text("How much is left?"))

    [trace] = get_traces()
    assert extract_request_from_trace(trace) == "How much is left?"
    assert extract_response_from_trace(trace) == "1,344 kcal left."
    assert _root(trace).span_type == SpanType.AGENT
    assert [call.name for call in extract_tools_called_from_trace(trace)] == ["get_remaining"]
    assert trace.info.trace_metadata["mlflow.trace.session"] == session_id


@pytest.mark.asyncio
async def test_thought_parts_are_not_in_the_reply():
    mlflow.google_adk.autolog()
    runner = InMemoryRunner(
        agent=_agent([[_text("I will say 2000.", thought=True), _text("Your goal is 1800 kcal.")]])
    )

    await _send(runner, await _session(runner), _text("What is my goal?"))

    [trace] = get_traces()
    assert extract_response_from_trace(trace) == "Your goal is 1800 kcal."


@pytest.mark.asyncio
async def test_final_texts_from_several_agents_are_joined():
    mlflow.google_adk.autolog()
    agent = SequentialAgent(
        name="flow",
        sub_agents=[
            LlmAgent(name="greeter", model=ScriptedModel(model="s", replies=[[_text("Hi.")]])),
            LlmAgent(name="coach", model=ScriptedModel(model="s", replies=[[_text("1800 kcal.")]])),
        ],
    )
    runner = InMemoryRunner(agent=agent)

    await _send(runner, await _session(runner), _text("What is my goal?"))

    [trace] = get_traces()
    assert extract_response_from_trace(trace) == "Hi.\n1800 kcal."


@pytest.mark.asyncio
@pytest.mark.parametrize(("confirmed", "label"), [(True, "accepted"), (False, "declined")])
async def test_confirmation_card_turns_are_labelled(confirmed, label):
    mlflow.google_adk.autolog()
    tool = FunctionTool(log_food, require_confirmation=True)
    runner = InMemoryRunner(
        agent=_agent([[_call("log_food", food="egg", qty=2)], [_text("Done.")]], tools=[tool])
    )
    session_id = await _session(runner)

    events = await _send(runner, session_id, _text("Log 2 eggs"))
    [card] = [
        call
        for event in events
        for call in event.get_function_calls()
        if call.name == "adk_request_confirmation"
    ]
    answer = types.FunctionResponse(id=card.id, name=card.name, response={"confirmed": confirmed})
    await _send(runner, session_id, types.Part(function_response=answer))

    shown, answered = get_traces()[::-1]
    assert extract_response_from_trace(shown) == "[confirmation requested: log_food]"
    assert extract_request_from_trace(answered) == f"[confirmation {label}]"


@pytest.mark.asyncio
async def test_agent_tool_run_stays_in_the_callers_session():
    mlflow.google_adk.autolog()
    helper = LlmAgent(name="helper", model=ScriptedModel(model="s", replies=[[_text("1800 kcal")]]))
    runner = InMemoryRunner(
        agent=_agent(
            [[_call("helper", request="What is my goal?")], [_text("Your goal is 1800 kcal.")]],
            tools=[AgentTool(agent=helper)],
        )
    )
    session_id = await _session(runner)

    await _send(runner, session_id, _text("Ask the helper for my goal"))

    [trace] = get_traces()
    assert trace.info.trace_metadata["mlflow.trace.session"] == session_id
    assert extract_request_from_trace(trace) == "Ask the helper for my goal"
    assert extract_response_from_trace(trace) == "Your goal is 1800 kcal."


@pytest.mark.asyncio
async def test_autolog_disable_prevents_new_traces():
    mlflow.google_adk.autolog()
    mlflow.google_adk.autolog(disable=True)
    runner = InMemoryRunner(agent=_agent([[_text("Hi.")]]))

    await _send(runner, await _session(runner), _text("Hello"))

    assert get_traces() == []
