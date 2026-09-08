from __future__ import annotations

import argparse
import json
import sys

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import NO_OP_SPAN_TRACE_ID

_TOOL_CALL_ID = "call_weather_sydney_001"
_TOOL_INPUT = {"location": "Sydney"}
_TOOL_OUTPUT = {
    "location": "Sydney",
    "temperature_c": 22,
    "conditions": "Sunny",
    "synthetic": True,
}


def emit_synthetic_trace(tracking_uri: str, experiment_id: str | None = None) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    if experiment_id:
        mlflow.set_experiment(experiment_id=experiment_id)

    tool_arguments = json.dumps(_TOOL_INPUT, separators=(",", ":"))
    tool_result = json.dumps(_TOOL_OUTPUT, separators=(",", ":"))
    messages = [
        {"role": "user", "content": "What's the weather in Sydney right now?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": _TOOL_CALL_ID,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": tool_arguments},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": _TOOL_CALL_ID,
            "content": tool_result,
        },
    ]
    output = {
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

    with mlflow.start_span("weather_agent", span_type=SpanType.AGENT) as root_span:
        root_span.set_inputs({"messages": messages})
        with mlflow.start_span("get_weather", span_type=SpanType.TOOL) as tool_span:
            tool_span.set_inputs(_TOOL_INPUT)
            tool_span.set_outputs(_TOOL_OUTPUT)
        root_span.set_outputs(output)

    mlflow.flush_trace_async_logging()
    if root_span.trace_id == NO_OP_SPAN_TRACE_ID:
        raise RuntimeError("Synthetic trace was not emitted because MLflow tracing is disabled.")
    return root_span.trace_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-id")
    args = parser.parse_args(argv)
    sys.stdout.write(f"{emit_synthetic_trace(args.tracking_uri, args.experiment_id)}\n")


if __name__ == "__main__":
    main()
