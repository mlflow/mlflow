import json
import logging
from typing import TYPE_CHECKING, Any, Callable

from opentelemetry.trace import NoOpTracer
from pydantic import BaseModel

import mlflow
from mlflow.entities.span import Span, SpanType
from mlflow.entities.trace import Trace
from mlflow.genai.evaluation.utils import is_none_or_nan
from mlflow.genai.utils.data_validation import check_model_prediction
from mlflow.models.evaluation.utils.trace import configure_autologging_for_evaluation
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracing.display.display_handler import IPythonTraceDisplayHandler
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.tracking.client import MlflowClient

if TYPE_CHECKING:
    from mlflow.genai.evaluation.entities import EvalItem

_logger = logging.getLogger(__name__)

_MESSAGE_KEY = "message"
_MESSAGES_KEY = "messages"
_CHOICES_KEY = "choices"
_CONTENT_KEY = "content"


def convert_predict_fn(predict_fn: Callable[..., Any], sample_input: Any) -> Callable[..., Any]:
    """
    Check the predict_fn is callable and add trace decorator if it is not already traced.
    """
    with (
        NoOpTracerPatcher() as counter,
        # Enable auto-tracing before checking if the predict_fn produces traces, so that
        # functions using auto-traceable libraries (OpenAI, LangChain, etc.) are correctly
        # identified as traced functions
        configure_autologging_for_evaluation(enable_tracing=True),
    ):
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


def parse_inputs_to_str(value: Any) -> str:
    """Parse the inputs to a string compatible with the judges API"""
    if is_none_or_nan(value):
        # The DBX managed backend doesn't allow empty inputs. This is
        # a temporary workaround to bypass the validation.
        return " "
    if isinstance(value, str):
        return value

    value = _to_dict(value)

    # Special handling for "messages" key.
    if (messages := value.get(_MESSAGES_KEY)) and len(messages) > 0:
        contents = [m.get(_CONTENT_KEY) for m in messages]
        # If the message contains multiple messages, dump the whole messages object.
        if len(contents) > 1 and all(isinstance(c, str) for c in contents):
            return json.dumps(messages)
        # If the message contains a single message, return the content.
        elif isinstance(contents[-1], str):
            return contents[-1]
    return str(value)


def parse_outputs_to_str(value: Any) -> str:
    """Parse the outputs to a string compatible with the judges API"""
    if is_none_or_nan(value):
        return " "
    if isinstance(value, str):
        return value

    # PyFuncModel.predict wraps the output in a list
    if isinstance(value, list) and len(value) > 0:
        return parse_outputs_to_str(value[0])

    value = _to_dict(value)

    # Special handling for chat response
    if _is_chat_choices(value.get(_CHOICES_KEY)):
        content = value[_CHOICES_KEY][0][_MESSAGE_KEY][_CONTENT_KEY]
    elif _is_chat_messages(value.get(_MESSAGES_KEY)):
        content = value[_MESSAGES_KEY][-1][_CONTENT_KEY]
    else:
        content = json.dumps(value, cls=TraceJSONEncoder)
    return content


def _is_chat_choices(maybe_choices: Any) -> bool:
    if (
        not maybe_choices
        or not isinstance(maybe_choices, list)
        or not isinstance(maybe_choices[0], dict)
    ):
        return False

    message = maybe_choices[0].get(_MESSAGE_KEY)
    return _is_chat_messages([message])


def _is_chat_messages(maybe_messages: Any) -> bool:
    return (
        maybe_messages
        and len(maybe_messages) > 0
        and isinstance(maybe_messages[-1], dict)
        and isinstance(maybe_messages[-1].get(_CONTENT_KEY), str)
    )


def _to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()

    if isinstance(obj, BaseModel):
        return obj.model_dump()

    # Convert to JSON string and then back to dictionary to handle nested objects
    json_str = json.dumps(obj, cls=TraceJSONEncoder)
    return json.loads(json_str)


def extract_retrieval_context_from_trace(trace: Trace | None) -> dict[str, list[Any]]:
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


def _parse_chunk(chunk: Any) -> dict[str, Any] | None:
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


def create_minimal_trace(eval_item: "EvalItem") -> Trace:
    """
    Create a minimal trace object with a single span, based on given inputs/outputs.
    """
    from mlflow.pyfunc.context import Context, set_prediction_context

    # Set the context so that the trace is logged synchronously
    context = Context(request_id=eval_item.request_id, is_evaluate=True)
    with set_prediction_context(context):
        with mlflow.start_span(name="root_span", span_type=SpanType.CHAIN) as root_span:
            root_span.set_inputs(eval_item.inputs)
            root_span.set_outputs(eval_item.outputs)
        return mlflow.get_trace(root_span.trace_id)
