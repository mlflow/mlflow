import json

import opentelemetry.trace as trace_api
from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.telemetry.tracer import get_tracer

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces


def _build_agent_result(text: str, in_tokens: int = 1, out_tokens: int = 1):
    metrics = EventLoopMetrics()
    metrics.accumulated_usage["inputTokens"] = in_tokens
    metrics.accumulated_usage["outputTokens"] = out_tokens
    metrics.accumulated_usage["totalTokens"] = in_tokens + out_tokens
    message = {"role": "assistant", "content": [{"text": text}]}
    return message, AgentResult("end_turn", message, metrics, state={})


def test_strands_autolog_single_trace():
    mlflow.strands.autolog()

    tracer = get_tracer()
    user_msg = {"role": "user", "content": [{"text": "hello"}]}
    _, result = _build_agent_result("hi", 1, 2)
    span = tracer.start_agent_span(message=user_msg, agent_name="agent")
    tracer.end_agent_span(span, response=result)

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

    tracer = get_tracer()
    user_msg = {"role": "user", "content": [{"text": "add numbers"}]}
    _, result = _build_agent_result("3", 1, 1)

    agent_span = tracer.start_agent_span(message=user_msg, agent_name="agent")
    tool_use = {"toolUseId": "tool-1", "name": "sum", "input": {"a": 1, "b": 2}}
    tool_span = tracer.start_tool_call_span(tool_use, parent_span=agent_span)
    tool_result = {"toolUseId": "tool-1", "status": "success", "content": [{"json": 3}]}
    tracer.end_tool_call_span(tool_span, tool_result=tool_result)
    tracer.end_agent_span(agent_span, response=result)

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

    tracer = get_tracer()
    msg1 = {"role": "user", "content": [{"text": "add numbers"}]}
    _, res1 = _build_agent_result("3", 1, 1)
    agent1_span = tracer.start_agent_span(message=msg1, agent_name="agent1")
    tool_use = {"toolUseId": "tool-1", "name": "sum", "input": {"a": 1, "b": 2}}
    tool_span = tracer.start_tool_call_span(tool_use, parent_span=agent1_span)
    tool_result = {"toolUseId": "tool-1", "status": "success", "content": [{"json": 3}]}
    tracer.end_tool_call_span(tool_span, tool_result=tool_result)

    msg2 = {"role": "user", "content": [{"text": "hello"}]}
    _, res2 = _build_agent_result("hi", 1, 1)
    with trace_api.use_span(agent1_span, end_on_exit=False):
        agent2_span = tracer.start_agent_span(message=msg2, agent_name="agent2")
        tracer.end_agent_span(agent2_span, response=res2)

    tracer.end_agent_span(agent1_span, response=res1)

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

    tracer = get_tracer()
    user_msg = {"role": "user", "content": [{"text": "hello"}]}
    second_msg = [{"text": "hi again"}]
    _, result = _build_agent_result("hi", 1, 2)
    span = tracer.start_agent_span(message=user_msg, agent_name="agent")
    span.add_event("gen_ai.user.message", {"content": json.dumps(second_msg)})
    tracer.end_agent_span(span, response=result)

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.inputs == [
        {"role": "user", "content": [{"text": "hello"}]},
        {"role": "user", "content": second_msg},
    ]
