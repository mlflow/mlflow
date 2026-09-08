from __future__ import annotations

import json

import mlflow
from mlflow.agent.setup._synthetic_trace import emit_synthetic_trace
from mlflow.entities import SpanType


def test_emit_synthetic_trace(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment("synthetic-trace-test")

    trace_id = emit_synthetic_trace(tracking_uri, experiment_id)
    trace = mlflow.get_trace(trace_id, flush=True)

    assert trace.info.experiment_id == experiment_id
    assert len(trace.data.spans) == 2
    root_span, tool_span = trace.data.spans
    assert root_span.name == "weather_agent"
    assert root_span.span_type == SpanType.AGENT
    assert root_span.parent_id is None
    assert tool_span.name == "get_weather"
    assert tool_span.span_type == SpanType.TOOL
    assert tool_span.parent_id == root_span.span_id
    assert tool_span.inputs == {"location": "Sydney"}
    assert tool_span.outputs == {
        "location": "Sydney",
        "temperature_c": 22,
        "conditions": "Sunny",
        "synthetic": True,
    }

    messages = root_span.inputs["messages"]
    assert messages[0] == {
        "role": "user",
        "content": "What's the weather in Sydney right now?",
    }
    tool_call = messages[1]["tool_calls"][0]
    assert messages[1]["content"] is None
    assert tool_call == {
        "id": "call_weather_sydney_001",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"location":"Sydney"}'},
    }
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == tool_call["id"]
    assert json.loads(messages[2]["content"]) == tool_span.outputs
    assert root_span.outputs == {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "Sydney is 22°C and sunny in this synthetic setup example. "
                        "This is not live weather data."
                    ),
                }
            }
        ]
    }
