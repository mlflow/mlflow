import json
import logging
from typing import Any, Callable, Optional

from opentelemetry.trace import NoOpTracer

import mlflow
from mlflow.entities.span import Span, SpanType
from mlflow.entities.trace import Trace
from mlflow.genai.utils.data_validation import check_model_prediction
from mlflow.tracing.utils import TraceJSONEncoder

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
    # If it is a single key dictionary, extract the value
    if isinstance(inputs, dict) and len(inputs) == 1:
        inputs = list(inputs.values())[0]

    return inputs if isinstance(inputs, str) else json.dumps(inputs, default=TraceJSONEncoder)


def parse_output_to_str(output: Any) -> str:
    """Parse the output to a string compatible with the judges API"""
    return output if isinstance(output, str) else json.dumps(output, default=TraceJSONEncoder)


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
