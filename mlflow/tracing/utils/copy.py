from typing import Any

from mlflow.entities.span import LiveSpan, Span
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_STATE
from mlflow.tracing.trace_manager import InMemoryTraceManager


def copy_trace_to_experiment(trace_dict: dict[str, Any], experiment_id: str | None = None) -> str:
    """
    Copy the given trace to the current experiment.
    The copied trace will have a new trace ID and location metadata.

    Args:
        trace_dict: The trace dictionary returned from model serving endpoint.
            This can be either V2 or V3 trace.
        experiment_id: The ID of the experiment to copy the trace to.
            If not provided, the trace will be copied to the current experiment.
    """
    new_trace_id = None
    new_root_span = None
    trace_manager = InMemoryTraceManager.get_instance()
    spans = [Span.from_dict(span_dict) for span_dict in trace_dict["data"]["spans"]]

    # Create a copy of spans in the current experiment
    for old_span in spans:
        new_span = LiveSpan.from_immutable_span(
            span=old_span,
            parent_span_id=old_span.parent_id,
            trace_id=new_trace_id,
            # Only set the experiment ID for the root span.
            experiment_id=experiment_id if old_span.parent_id is None else None,
        )
        # we need to register the span to trace manager first before ending it
        # otherwise the span will not be correctly exported
        trace_manager.register_span(new_span)
        if old_span.parent_id is None:
            new_root_span = new_span
            new_trace_id = new_span.trace_id
        else:
            new_span.end(end_time_ns=old_span.end_time_ns)

    if new_trace_id is None:
        raise MlflowException(
            "Root span not found in the trace. Perhaps the trace data is corrupted.",
            error_code=INVALID_STATE,
        )

    if info := trace_dict.get("info"):
        with trace_manager.get_trace(trace_id=new_trace_id) as trace:
            # Copy user tags (excluding mlflow internal tags)
            if all_tags := info.get("tags"):
                if user_tags := {k: v for k, v in all_tags.items() if not k.startswith("mlflow.")}:
                    trace.info.tags.update(user_tags)

            # Copy trace metadata
            if trace_metadata := info.get("trace_metadata"):
                trace.info.trace_metadata.update(trace_metadata)

    # Close the root span triggers the trace export.
    new_root_span.end(end_time_ns=spans[0].end_time_ns)
    return new_trace_id
