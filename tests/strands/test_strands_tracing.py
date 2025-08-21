import json

import opentelemetry.trace as trace_api
from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.models.model import Model
from strands.telemetry.metrics import EventLoopMetrics

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces


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
    return message, AgentResult("end_turn", message, metrics, state={})


def test_strands_autolog_single_trace():
    mlflow.strands.autolog()

    agent = Agent(model=_DummyModel(), name="agent")
    user_msg = {"role": "user", "content": [{"text": "hello"}]}
    _, result = _build_agent_result("hi", 1, 2)
    agent.trace_span = agent._start_agent_trace_span(message=user_msg)
    agent._end_agent_trace_span(response=result)

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


def test_function_calling_creates_single_trace():
    mlflow.strands.autolog()

    agent = Agent(model=_DummyModel(), name="agent")
    user_msg = {"role": "user", "content": [{"text": "add numbers"}]}
    _, result = _build_agent_result("3", 1, 1)
    agent.trace_span = agent._start_agent_trace_span(message=user_msg)
    tool_use = {"toolUseId": "tool-1", "name": "sum", "input": {"a": 1, "b": 2}}
    tool_span = agent.tracer.start_tool_call_span(tool_use, parent_span=agent.trace_span)
    tool_result = {"toolUseId": "tool-1", "status": "success", "content": [{"json": 3}]}
    agent.tracer.end_tool_call_span(tool_span, tool_result=tool_result)
    agent._end_agent_trace_span(response=result)

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert spans[0].span_type == SpanType.AGENT
    assert spans[1].span_type == SpanType.TOOL
    assert spans[0].inputs == [{"role": "user", "content": [{"text": "add numbers"}]}]
    assert spans[0].outputs == 3
    assert spans[1].inputs == [{"role": "tool", "content": {"a": 1, "b": 2}}]
    assert spans[1].outputs == [{"json": 3}]


def test_multiple_agents_single_trace():
    mlflow.strands.autolog()

    agent1 = Agent(model=_DummyModel(), name="agent1")
    msg1 = {"role": "user", "content": [{"text": "add numbers"}]}
    _, res1 = _build_agent_result("3", 1, 1)
    agent1.trace_span = agent1._start_agent_trace_span(message=msg1)
    tool_use = {"toolUseId": "tool-1", "name": "sum", "input": {"a": 1, "b": 2}}
    tool_span = agent1.tracer.start_tool_call_span(tool_use, parent_span=agent1.trace_span)
    tool_result = {"toolUseId": "tool-1", "status": "success", "content": [{"json": 3}]}
    agent1.tracer.end_tool_call_span(tool_span, tool_result=tool_result)

    agent2 = Agent(model=_DummyModel(), name="agent2")
    msg2 = {"role": "user", "content": [{"text": "hello"}]}
    _, res2 = _build_agent_result("hi", 1, 1)
    with trace_api.use_span(agent1.trace_span, end_on_exit=False):
        agent2.trace_span = agent2._start_agent_trace_span(message=msg2)
        agent2._end_agent_trace_span(response=res2)

    agent1._end_agent_trace_span(response=res1)

    traces = get_traces()
    assert len(traces) == 1
    spans = traces[0].data.spans
    assert spans[0].span_type == SpanType.AGENT
    assert spans[0].inputs == [{"role": "user", "content": [{"text": "add numbers"}]}]
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
    user_msg = {"role": "user", "content": [{"text": "hello"}]}
    second_msg = [{"text": "hi again"}]
    _, result = _build_agent_result("hi", 1, 2)
    agent.trace_span = agent._start_agent_trace_span(message=user_msg)
    agent.trace_span.add_event("gen_ai.user.message", {"content": json.dumps(second_msg)})
    agent._end_agent_trace_span(response=result)

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.inputs == [
        {"role": "user", "content": [{"text": "hello"}]},
        {"role": "user", "content": second_msg},
    ]


def test_autolog_disable():
    from strands.telemetry.tracer import get_tracer

    mlflow.strands.autolog(disable=True)

    tracer = get_tracer()
    assert not hasattr(tracer, "span_processor")


def test_autolog_disable_autolog():
    mlflow.strands.autolog()

    agent = Agent(model=_DummyModel(), name="agent")
    user_msg = {"role": "user", "content": [{"text": "hello"}]}
    _, result = _build_agent_result("hi", 1, 1)
    agent.trace_span = agent._start_agent_trace_span(message=user_msg)
    agent._end_agent_trace_span(response=result)

    traces = get_traces()
    assert len(traces) == 1

    mlflow.strands.autolog(disable=True)

    agent2 = Agent(model=_DummyModel(), name="agent2")
    msg2 = {"role": "user", "content": [{"text": "bye"}]}
    _, result2 = _build_agent_result("cya", 1, 1)
    agent2.trace_span = agent2._start_agent_trace_span(message=msg2)
    agent2._end_agent_trace_span(response=result2)

    assert len(get_traces()) == 1
