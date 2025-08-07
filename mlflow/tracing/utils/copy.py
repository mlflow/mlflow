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
            experiment_id=experiment_id,
            # Don't close the root span until the end so that we only export the trace
            # after all spans are copied.
            end_trace=old_span.parent_id is not None,
        )
        trace_manager.register_span(new_span)
        if old_span.parent_id is None:
            new_root_span = new_span
            new_trace_id = new_span.trace_id

    if new_trace_id is None:
        raise MlflowException(
            "Root span not found in the trace. Perhaps the trace data is corrupted.",
            error_code=INVALID_STATE,
        )

    user_tags = {k: v for k, v in trace_dict["info"]["tags"].items() if not k.startswith("mlflow.")}
    with trace_manager.get_trace(trace_id=new_trace_id) as trace:
        trace.info.tags.update(user_tags)

    # Close the root span triggers the trace export.
    new_root_span.end(end_time_ns=spans[0].end_time_ns)
    return new_trace_id
