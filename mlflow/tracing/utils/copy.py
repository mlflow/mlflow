from typing import Any

import mlflow
from mlflow.entities.span import LiveSpan, Span
from mlflow.tracing.trace_manager import InMemoryTraceManager


def copy_trace_to_current_experiment(trace_dict: dict[str, Any]) -> str:
    """
    Copy the given trace to the current experiment.
    The copied trace will have a new trace ID and location metadata.

    Args:
        trace_dict: The trace dictionary returned from model serving endpoint.
            This can be either V2 or V3 trace.
    """
    new_trace_id, new_root_span = None, None
    spans = [Span.from_dict(span_dict) for span_dict in trace_dict["data"]["spans"]]

    # Create a copy of spans in the current experiment
    for old_span in spans:
        new_span = LiveSpan.from_immutable_span(
            span=old_span,
            parent_span_id=old_span.parent_id,
            trace_id=new_trace_id,
            # Don't close the root span until the end so that we only export the trace
            # after all spans are copied.
            end_trace=old_span.parent_id is not None,
        )
        InMemoryTraceManager.get_instance().register_span(new_span)
        if old_span.parent_id is None:
            new_root_span = new_span
            new_trace_id = new_span.trace_id

    user_tags = {k: v for k, v in trace_dict["info"]["tags"].items() if not k.startswith("mlflow.")}
    if user_tags:
        mlflow.update_current_trace(tags=user_tags)

    # Close the root span triggers the trace export.
    new_root_span.end(end_time_ns=spans[0].end_time_ns)
    return new_trace_id
