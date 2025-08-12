from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.telemetry.tracer import get_tracer

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces


def test_strands_autolog_records_span():
    mlflow.strands.autolog()
    tracer = get_tracer()
    message = {"role": "user", "content": [{"text": "hello"}]}
    metrics = EventLoopMetrics()
    metrics.accumulated_usage["inputTokens"] = 1
    metrics.accumulated_usage["outputTokens"] = 2
    metrics.accumulated_usage["totalTokens"] = 3
    result_message = {"role": "assistant", "content": [{"text": "hi"}]}
    result = AgentResult("end_turn", result_message, metrics, state={})
    span = tracer.start_agent_span(message=message, agent_name="agent")
    tracer.end_agent_span(span, response=result)
    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.span_type == SpanType.AGENT
    assert span.inputs == {"content": [{"text": "hello"}]}
    assert span.outputs == "hi\n"
    assert span.attributes[SpanAttributeKey.CHAT_USAGE] == {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
    }
