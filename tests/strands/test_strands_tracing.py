import json
from unittest.mock import patch

from strands import Agent
from strands.agent.agent import run_tool
from strands.agent.agent_result import AgentResult
from strands.models.model import Model
from strands.telemetry.metrics import EventLoopMetrics, Trace
from strands.tools.executor import run_tools
from strands.tools.tools import PythonAgentTool

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces


async def sum_tool(tool_use, **_):
    a = tool_use["input"]["a"]
    b = tool_use["input"]["b"]
    return {
        "toolUseId": tool_use["toolUseId"],
        "status": "success",
        "content": [{"json": a + b}],
    }


tool = PythonAgentTool(
    "sum",
    {
        "name": "sum",
        "description": "add numbers 1 2",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    sum_tool,
)


class _DummyModel(Model):
    def __init__(self):
        self.config = {}

    def update_config(self, **model_config):
        self.config.update(model_config)

    def get_config(self):
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        if False:
            yield {}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if False:
            yield {}


def _build_agent_result(text: str, in_tokens: int = 1, out_tokens: int = 1):
    metrics = EventLoopMetrics()
    metrics.accumulated_usage["inputTokens"] = in_tokens
    metrics.accumulated_usage["outputTokens"] = out_tokens
    metrics.accumulated_usage["totalTokens"] = in_tokens + out_tokens
    message = {"role": "assistant", "content": [{"text": text}]}
    return AgentResult("end_turn", message, metrics, state={})


def test_strands_autolog_single_trace():
    mlflow.strands.autolog()

    agent = Agent(model=_DummyModel(), name="agent")
    result = _build_agent_result("hi", 1, 2)

    async def new_run_loop(self, message, invocation_state):
        yield {"stop": (result.stop_reason, result.message, result.metrics, result.state)}

    with patch.object(Agent, "_run_loop", new_run_loop):
        agent("hello")

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.span_type == SpanType.AGENT
    assert span.inputs == [{"role": "user", "content": [{"text": "hello"}]}]
    assert span.outputs == "hi\n"
    assert span.attributes[SpanAttributeKey.CHAT_USAGE] == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }

    mlflow.strands.autolog(disable=True)
    with patch.object(Agent, "_run_loop", new_run_loop):
        agent("bye")
        assert len(get_traces()) == 1


def test_function_calling_creates_single_trace():
    mlflow.strands.autolog()

    agent = Agent(model=_DummyModel(), tools=[tool], name="agent")
    result = _build_agent_result("3", 1, 1)

    async def new_run_loop(self, message, invocation_state):
        tool_use = {"toolUseId": "tool-1", "name": "sum", "input": {"a": 1, "b": 2}}
        tool_events = run_tools(
            handler=lambda tool_use: run_tool(self, tool_use, invocation_state),
            tool_uses=[tool_use],
            event_loop_metrics=self.event_loop_metrics,
            invalid_tool_use_ids=[],
            tool_results=[],
            cycle_trace=Trace("cycle"),
            parent_span=self.trace_span,
        )
        async for _ in tool_events:
            pass
        yield {"stop": (result.stop_reason, result.message, result.metrics, result.state)}

    with patch.object(Agent, "_run_loop", new_run_loop):
        agent("add numbers 1 2 1 2")

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert spans[0].span_type == SpanType.AGENT
    assert spans[1].span_type == SpanType.TOOL
    assert spans[0].inputs == [{"role": "user", "content": [{"text": "add numbers 1 2 1 2"}]}]
    assert spans[0].outputs == 3
    assert spans[1].inputs == [{"role": "tool", "content": {"a": 1, "b": 2}}]
    assert spans[1].outputs == [{"json": 3}]


def test_multiple_agents_single_trace():
    mlflow.strands.autolog()

    agent1 = Agent(model=_DummyModel(), tools=[tool], name="agent1")
    agent2 = Agent(model=_DummyModel(), name="agent2")

    res1 = _build_agent_result("3", 1, 1)
    res2 = _build_agent_result("hi", 1, 1)

    async def new_run_loop(self, message, invocation_state):
        if self is agent1:
            tool_use = {"toolUseId": "tool-1", "name": "sum", "input": {"a": 1, "b": 2}}
            tool_events = run_tools(
                handler=lambda tool_use: run_tool(self, tool_use, invocation_state),
                tool_uses=[tool_use],
                event_loop_metrics=self.event_loop_metrics,
                invalid_tool_use_ids=[],
                tool_results=[],
                cycle_trace=Trace("cycle"),
                parent_span=self.trace_span,
            )
            async for _ in tool_events:
                pass
            await agent2.invoke_async("hello")
            yield {"stop": (res1.stop_reason, res1.message, res1.metrics, res1.state)}
        else:
            yield {"stop": (res2.stop_reason, res2.message, res2.metrics, res2.state)}

    with patch.object(Agent, "_run_loop", new_run_loop):
        agent1("add numbers 1 2")

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].inputs == [{"role": "user", "content": [{"text": "add numbers 1 2"}]}]
    assert spans[0].outputs == 3
    assert spans[1].span_type == SpanType.TOOL
    assert spans[1].inputs == [{"role": "tool", "content": {"a": 1, "b": 2}}]
    assert spans[1].outputs == [{"json": 3}]
    assert spans[2].span_type == SpanType.AGENT
    assert spans[2].inputs == [{"role": "user", "content": [{"text": "hello"}]}]
    assert spans[2].outputs == "hi\n"


def test_span_records_multiple_messages():
    mlflow.strands.autolog()

    agent = Agent(model=_DummyModel(), name="agent")
    result = _build_agent_result("hi", 1, 2)
    second_msg = [{"text": "hi again"}]

    async def new_run_loop(self, message, invocation_state):
        self.trace_span.add_event("gen_ai.user.message", {"content": json.dumps(second_msg)})
        yield {"stop": (result.stop_reason, result.message, result.metrics, result.state)}

    with patch.object(Agent, "_run_loop", new_run_loop):
        agent("hello")

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.inputs == [
        {"role": "user", "content": [{"text": "hello"}]},
        {"role": "user", "content": second_msg},
    ]


def test_autolog_disable_prevents_new_traces():
    mlflow.strands.autolog()

    agent1 = Agent(model=_DummyModel(), name="agent1")
    agent2 = Agent(model=_DummyModel(), name="agent2")

    res1 = _build_agent_result("hi", 1, 1)
    res2 = _build_agent_result("cya", 1, 1)

    async def new_run_loop(self, message, invocation_state):
        result = res1 if self is agent1 else res2
        yield {"stop": (result.stop_reason, result.message, result.metrics, result.state)}

    with patch.object(Agent, "_run_loop", new_run_loop):
        agent1("hello")
        assert len(get_traces()) == 1

    mlflow.strands.autolog(disable=True)
    with patch.object(Agent, "_run_loop", new_run_loop):
        agent2("bye")
        assert len(get_traces()) == 1
