import logging
from typing import Any, Callable, Optional

from opentelemetry.trace import NoOpTracer

import mlflow
from mlflow.entities.span import LiveSpan, Span, SpanType
from mlflow.entities.trace import Trace
from mlflow.genai.utils.data_validation import check_model_prediction
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracing.display.display_handler import IPythonTraceDisplayHandler
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracking.client import MlflowClient

_logger = logging.getLogger(__name__)


def convert_predict_fn(predict_fn: Callable, sample_input: Any) -> Callable:
    """
    Check the predict_fn is callable and add trace decorator if it is not already traced.
    """
    with NoOpTracerPatcher() as counter:
        check_model_prediction(predict_fn, sample_input)

    if counter.count == 0:
        predict_fn = mlflow.trace(predict_fn)

    # Wrap the prediction function to unwrap the inputs dictionary into keyword arguments.
    return lambda request: predict_fn(**request)


class NoOpTracerPatcher:
    """
    A context manager to count the number of times NoOpTracer's start_span is called.

    The check is done in the following steps so it doesn't have any side effects:
    1. Disable tracing.
    2. Patch the NoOpTracer.start_span method to count the number of times it is called.
        NoOpTracer is used when tracing is disabled.
    3. Call the predict function with the sample input.
    4. Restore the original NoOpTracer.start_span method and re-enable tracing.


    WARNING: This function is not thread-safe. We do not provide support for running
        `mlflow.genai.evaluate` in multi-threaded environments.`
    """

    def __init__(self):
        self.count = 0

    def __enter__(self):
        self.original = NoOpTracer.start_span

        def _patched_start_span(_self, *args, **kwargs):
            self.count += 1
            return self.original(_self, *args, **kwargs)

        NoOpTracer.start_span = _patched_start_span
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        NoOpTracer.start_span = self.original


def parse_inputs_to_str(inputs: Any) -> str:
    """Parse the inputs to a request string compatible with the judges API"""
    from databricks.rag_eval.utils import input_output_utils

    return input_output_utils.request_to_string(inputs)


def parse_output_to_str(output: Any) -> str:
    """Parse the output to a string compatible with the judges API"""
    from databricks.rag_eval.utils import input_output_utils

    return input_output_utils.response_to_string(output)


def extract_retrieval_context_from_trace(trace: Optional[Trace]) -> dict[str, list]:
    """
    Extract the retrieval context from the trace.
    Only consider the last retrieval span in the trace if there are multiple retrieval spans.
    If the trace does not have a retrieval span, return None.
    ⚠️ Warning: Please make sure to not throw exception. If fails, return None.
    """
    if trace is None or trace.data is None:
        return {}

    # Only consider the top-level retrieval spans
    top_level_retrieval_spans = _get_top_level_retrieval_spans(trace)
    if len(top_level_retrieval_spans) == 0:
        return {}

    retrieved = {}  # span_id -> list of documents

    for retrieval_span in top_level_retrieval_spans:
        try:
            contexts = [_parse_chunk(chunk) for chunk in retrieval_span.outputs or []]
            retrieved[retrieval_span.span_id] = [c for c in contexts if c is not None]
        except Exception as e:
            _logger.debug(
                f"Fail to get retrieval context from span: {retrieval_span}. Error: {e!r}"
            )

    return retrieved


def _get_top_level_retrieval_spans(trace: Trace) -> list[Span]:
    """
    Get the top-level retrieval spans in the trace.
    Top-level retrieval spans are retrieval spans that are not children of other retrieval spans.
    For example, given the following spans:
    - Span A (Chain)
      - Span B (Retriever)
        - Span C (Retriever)
      - Span D (Retriever)
        - Span E (LLM)
          - Span F (Retriever)
    Span B and Span D are top-level retrieval spans.
    Span C and Span F are NOT top-level because they are children of other retrieval spans.
    """
    top_level_retrieval_spans = []
    # Cache span_id -> span mapping for fast lookup
    all_spans = {span.span_id: span for span in trace.data.spans}
    for span in trace.search_spans(span_type=SpanType.RETRIEVER):
        # Check if this span is a child of another retrieval span
        parent_id = span.parent_id
        while parent_id:
            parent_span = all_spans.get(parent_id)
            if not parent_span:
                # Malformed trace
                _logger.debug(
                    f"Malformed trace: span {span} has parent span ID {parent_id}, "
                    "but the parent span is not found in the trace."
                )
                break

            if parent_span.span_type == SpanType.RETRIEVER:
                # This span is a child of another retrieval span
                break

            parent_id = parent_span.parent_id
        else:
            # If the loop completes without breaking, this is a top-level span
            top_level_retrieval_spans.append(span)

    return top_level_retrieval_spans


def _parse_chunk(chunk: Any) -> Optional[dict[str, Any]]:
    if not isinstance(chunk, dict):
        return None

    doc = {"content": chunk.get("page_content")}
    if doc_uri := chunk.get("metadata", {}).get("doc_uri"):
        doc["doc_uri"] = doc_uri
    return doc


def clean_up_extra_traces(run_id: str, start_time_ms: int):
    """
    Clean up noisy traces generated outside predict function.

    Evaluation run should only contain traces that is being evaluated or generated by the predict
    function. If not, the result will not show the correct list of traces.
    Sometimes, there are extra traces generated during the evaluation, for example, custom scorer
    code might generate traces. This function cleans up those noisy traces.

    TODO: This is not a fundamental solution. Ideally, evaluation result should be able to render
        correct result even if there are extra traces in the run.

    Args:
        run_id: The ID of the run to clean up.
        start_time_ms: The start time of the evaluation in milliseconds.
    """
    from mlflow.tracking.fluent import _get_experiment_id

    try:
        # Search for all traces generated during evaluation
        traces = mlflow.search_traces(
            run_id=run_id,
            # Not download spans for efficiency
            include_spans=False,
            # Limit to traces generated after evaluation time to ensure we will not
            # delete traces generated before evaluation.
            filter_string=f"trace.timestamp >= {start_time_ms}",
            return_type="list",
        )
        extra_trace_ids = [
            # Traces from predict function should always have the EVAL_REQUEST_ID tag
            trace.info.trace_id
            for trace in traces
            if TraceTagKey.EVAL_REQUEST_ID not in trace.info.tags
        ]
        if extra_trace_ids:
            _logger.debug(
                f"Found {len(extra_trace_ids)} extra traces generated during evaluation run. "
                "Deleting them."
            )
            MlflowClient().delete_traces(
                experiment_id=_get_experiment_id(), trace_ids=extra_trace_ids
            )
            # Avoid displaying the deleted trace in notebook cell output
            for trace_id in extra_trace_ids:
                IPythonTraceDisplayHandler.get_instance().traces_to_display.pop(trace_id, None)
        else:
            _logger.debug("No extra traces found during evaluation run.")
    except Exception as e:
        _logger.warning(
            f"Failed to clean up extra traces generated during evaluation. The "
            f"result page might not show the correct list of traces. Error: {e}"
        )


def copy_model_serving_trace_to_eval_run(trace_dict: dict[str, Any]):
    """
    Copy a trace returned from model serving endpoint to the evaluation run.
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

    # Close the root span triggers the trace export.
    new_root_span.end(end_time_ns=spans[0].end_time_ns)
