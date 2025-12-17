import asyncio
import functools
import inspect
import json
import logging
import math
from typing import TYPE_CHECKING, Any, Callable

from cachetools.func import cached
from opentelemetry.trace import NoOpTracer
from pydantic import BaseModel, Field

import mlflow
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.span import Span, SpanType
from mlflow.entities.trace import Trace
from mlflow.environment_variables import (
    MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT,
    MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING,
    MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION,
)
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import get_chat_completions_with_structured_output
from mlflow.genai.utils.data_validation import check_model_prediction
from mlflow.genai.utils.prompts.available_tools_extraction import (
    get_available_tools_extraction_prompts,
)
from mlflow.models.evaluation.utils.trace import configure_autologging_for_evaluation
from mlflow.tracing.constant import (
    AssessmentMetadataKey,
    SpanAttributeKey,
    TraceMetadataKey,
    TraceTagKey,
)
from mlflow.tracing.display import IPythonTraceDisplayHandler
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.tracing.utils.search import traces_to_df
from mlflow.tracking.client import MlflowClient
from mlflow.utils.uri import is_databricks_uri

if TYPE_CHECKING:
    import pandas as pd

    from mlflow.genai.evaluation.entities import EvalItem, EvalResult
    from mlflow.genai.utils.type import FunctionCall
    from mlflow.types.chat import ChatTool

_logger = logging.getLogger(__name__)

_MESSAGE_KEY = "message"
_MESSAGES_KEY = "messages"
_CHOICES_KEY = "choices"
_CONTENT_KEY = "content"


def extract_request_from_trace(trace: Trace) -> str | None:
    """
    Extract request text from an MLflow trace object.

    Args:
        trace: MLflow trace object

    Returns:
        Extracted request text as string, or None if no root span
    """
    root_span = trace.data._get_root_span()
    if root_span is None:
        return None
    return parse_inputs_to_str(root_span.inputs)


def extract_response_from_trace(trace: Trace) -> str | None:
    """
    Extract response text from an MLflow trace object.

    Args:
        trace: MLflow trace object

    Returns:
        Extracted response text as string, or None if no root span
    """
    root_span = trace.data._get_root_span()
    if root_span is None:
        return None
    return parse_outputs_to_str(root_span.outputs)


def extract_inputs_from_trace(trace: Trace) -> Any:
    """
    Extract inputs from the root span of an MLflow trace.

    Args:
        trace: MLflow trace object

    Returns:
        Inputs from the root span, or None if no root span or inputs
    """
    root_span = trace.data._get_root_span()
    if root_span and root_span.inputs is not None:
        return root_span.inputs
    return None


def extract_outputs_from_trace(trace: Trace) -> Any:
    """
    Extract outputs from the root span of an MLflow trace.

    Args:
        trace: MLflow trace object

    Returns:
        Outputs from the root span, or None if no root span or outputs
    """
    root_span = trace.data._get_root_span()
    if root_span and root_span.outputs is not None:
        return root_span.outputs
    return None


def resolve_inputs_from_trace(
    inputs: Any | None, trace: Trace, *, extract_if_none: bool = True
) -> Any | None:
    """
    Extract inputs from trace if not provided.

    Args:
        inputs: Input data to evaluate. If None, will be extracted from trace.
        trace: MLflow trace object containing the execution to evaluate.
        extract_if_none: If True, extract from trace when inputs is None. If False, only
                        return the provided inputs value. Defaults to True.

    Returns:
        The provided inputs if not None, otherwise extracted inputs from trace,
        or None if extraction fails.
    """
    if inputs is None and trace is not None and extract_if_none:
        try:
            return extract_inputs_from_trace(trace)
        except Exception as e:
            _logger.debug(f"Could not extract inputs from trace: {e}")
    return inputs


def resolve_outputs_from_trace(
    outputs: Any | None, trace: Trace, *, extract_if_none: bool = True
) -> Any | None:
    """
    Extract outputs from trace if not provided.

    Args:
        outputs: Output data to evaluate. If None, will be extracted from trace.
        trace: MLflow trace object containing the execution to evaluate.
        extract_if_none: If True, extract from trace when outputs is None. If False, only
                        return the provided outputs value. Defaults to True.

    Returns:
        The provided outputs if not None, otherwise extracted outputs from trace,
        or None if extraction fails.
    """
    if outputs is None and trace is not None and extract_if_none:
        try:
            return extract_outputs_from_trace(trace)
        except Exception as e:
            _logger.debug(f"Could not extract outputs from trace: {e}")
    return outputs


def _get_exception_from_span(span: Span) -> str | None:
    """
    Extract exception information from span events.

    Args:
        span: The span to check for exception events.

    Returns:
        A formatted string containing exception information if found, None otherwise.
    """
    exception_events = [event for event in span.events if event.name == "exception"]
    if not exception_events:
        return None

    exception_event = exception_events[0]
    attrs = exception_event.attributes

    exception_type = attrs.get("exception.type", "Exception")

    if exception_message := attrs.get("exception.message"):
        return f"{exception_type}: {exception_message}"
    return exception_type


def extract_tools_called_from_trace(trace: Trace) -> list["FunctionCall"]:
    """
    Extract tool call information from TOOL type spans in a trace.

    This function extracts tool spans (spans with span_type==SpanType.TOOL) from a trace
    and returns them as a list of FunctionCall objects containing the tool name, inputs,
    and outputs.

    Args:
        trace: A single Trace object to extract tool calls from.

    Returns:
        List of FunctionCall objects.
        Returns empty list if no tool spans are found.

    Example:
        >>> trace = mlflow.get_trace(trace_id)
        >>> tools = extract_tools_called_from_trace(trace)
        >>> # Returns: [FunctionCall(name="tool_name", arguments={...}, outputs={...})]
    """
    from mlflow.genai.utils.type import FunctionCall

    tools_called = []
    tool_spans = trace.search_spans(span_type=SpanType.TOOL)

    for tool_span in sorted(tool_spans, key=lambda s: s.start_time_ns or 0):
        tool_info = FunctionCall(
            name=tool_span.name,
            arguments=tool_span.inputs or None,
            outputs=tool_span.outputs or None,
            exception=_get_exception_from_span(tool_span),
        )
        tools_called.append(tool_info)

    return tools_called


def parse_tool_call_messages_from_trace(trace: Trace) -> list[dict[str, str]]:
    """
    Extract and format tool call information from TOOL type spans in a trace.

    This function extracts tool spans (spans with span_type==SpanType.TOOL) from a trace
    and formats them as conversation messages with role='tool'. Each tool message includes
    the tool name, inputs, and outputs.

    Args:
        trace: A single Trace object to extract tool calls from.

    Returns:
        List of tool call messages in the format [{"role": "tool", "content": str}].
        Tool content includes the tool name, inputs, and outputs formatted as a string.
        Returns empty list if no tool spans are found.

    Example:
        >>> trace = mlflow.get_trace(trace_id)
        >>> tool_messages = parse_tool_call_messages_from_trace(trace)
        >>> # Returns: [{"role": "tool", "content": "Tool: name\\nInputs: ...\\nOutputs: ..."}]
    """
    tools_called = extract_tools_called_from_trace(trace)

    tool_messages = []
    for tool in tools_called:
        tool_info = f"Tool: {tool.name}"
        if tool.arguments is not None:
            tool_info += f"\nInputs: {tool.arguments}"
        if tool.outputs is not None:
            tool_info += f"\nOutputs: {tool.outputs}"
        if tool.exception is not None:
            tool_info += f"\nException: {tool.exception}"
        tool_messages.append({"role": "tool", "content": tool_info})

    return tool_messages


def resolve_conversation_from_session(
    session: list[Trace],
    *,
    include_tool_calls: bool = False,
) -> list[dict[str, str]]:
    """
    Extract conversation history from traces in session.

    Args:
        session: List of traces from the same session.
        include_tool_calls: If True, include tool call information from TOOL type spans
                           in the conversation. Default is False for backward compatibility.

    Returns:
        List of conversation messages in the format:
        [{"role": "user"|"assistant"|"tool", "content": str}].
        Each trace contributes user input and assistant output messages.
        If include_tool_calls is True, tool call messages (with inputs/outputs)
        are also included in chronological order.
    """
    # Sort traces by creation time (timestamp_ms)
    sorted_traces = sorted(session, key=lambda t: t.info.timestamp_ms)

    conversation = []
    for trace in sorted_traces:
        # Extract and parse input (user message)
        if inputs := extract_inputs_from_trace(trace):
            user_content = parse_inputs_to_str(inputs)
            if user_content and user_content.strip():
                conversation.append({"role": "user", "content": user_content})

        # Extract tool calls from TOOL type spans (if requested)
        if include_tool_calls:
            tool_messages = parse_tool_call_messages_from_trace(trace)
            conversation.extend(tool_messages)

        # Extract and parse output (assistant message)
        if outputs := extract_outputs_from_trace(trace):
            assistant_content = parse_outputs_to_str(outputs)
            if assistant_content and assistant_content.strip():
                conversation.append({"role": "assistant", "content": assistant_content})

    return conversation


def resolve_expectations_from_trace(
    expectations: dict[str, Any] | None,
    trace: Trace,
    source: AssessmentSourceType = AssessmentSourceType.HUMAN,
    *,
    extract_if_none: bool = True,
) -> dict[str, Any] | None:
    """
    Extract expectations from trace if not provided.

    Args:
        expectations: Dictionary of expected outcomes. If None, will be extracted from trace.
        trace: MLflow trace object containing the execution to evaluate.
        source: Assessment source type to filter expectations by. Defaults to HUMAN.
        extract_if_none: If True, extract from trace when expectations is None. If False, only
                        return the provided expectations value. Defaults to True.

    Returns:
        The provided expectations if not None, otherwise extracted expectations from trace,
        or None if extraction fails.
    """
    if expectations is None and trace is not None and extract_if_none:
        try:
            return extract_expectations_from_trace(trace, source=source)
        except Exception as e:
            _logger.debug(f"Could not extract expectations from trace: {e}")
    return expectations


def extract_expectations_from_trace(
    trace: Trace, source: str | None = None
) -> dict[str, Any] | None:
    """
    Extract expectations from trace assessments.

    Args:
        trace: MLflow trace object
        source: If specified, only extract expectations from the given source type.
                Must be one of the valid AssessmentSourceType values
                If None, extract all expectations regardless of source.

    Returns:
        Dictionary of expectations, or None if no expectations found
    """
    validated_source = AssessmentSourceType._standardize(source) if source is not None else None

    expectation_assessments = trace.search_assessments(type="expectation")

    if validated_source is not None:
        expectation_assessments = [
            exp
            for exp in expectation_assessments
            if exp.source and exp.source.source_type == validated_source
        ]

    if not expectation_assessments:
        return None

    return {exp.name: exp.expectation.value for exp in expectation_assessments}


def _wrap_async_predict_fn(async_fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap an async function to make it synchronous using asyncio.run with timeout.

    Args:
        async_fn: The async function to wrap

    Returns:
        A synchronous wrapper function that calls the async function with timeout
    """
    timeout = MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT.get()

    @functools.wraps(async_fn)
    def sync_wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            try:
                import nest_asyncio

                nest_asyncio.apply()
            except ImportError:
                raise MlflowException(
                    "Detected a running event loop (e.g., in Jupyter notebook). "
                    "To use async predict functions in notebook environments, "
                    "install nest-asyncio: pip install nest-asyncio"
                )

        return asyncio.run(asyncio.wait_for(async_fn(*args, **kwargs), timeout=timeout))

    return sync_wrapper


def convert_predict_fn(predict_fn: Callable[..., Any], sample_input: Any) -> Callable[..., Any]:
    """
    Check the predict_fn is callable and add trace decorator if it is not already traced.
    If the predict_fn is an async function, wrap it to make it synchronous.
    """
    # Detect if predict_fn is an async function and wrap it
    if inspect.iscoroutinefunction(predict_fn):
        _logger.debug(
            f"Detected async predict_fn. Wrapping with asyncio.run() with timeout of "
            f"{MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT.get()} seconds."
        )
        predict_fn = _wrap_async_predict_fn(predict_fn)
    if not MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION.get():
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


def is_none_or_nan(value: Any) -> bool:
    """
    Checks whether a value is None or NaN.

    NB: This function does not handle pandas.NA.
    """
    # isinstance(value, float) check is needed to ensure that math.isnan is not called on an array.
    return value is None or (isinstance(value, float) and math.isnan(value))


def _is_empty(value: Any) -> bool:
    """
    Check if a value is empty (None, empty dict, empty list, empty string, etc.).
    """
    if value is None:
        return True
    if isinstance(value, (dict, list, str)):
        return len(value) == 0
    return False


def parse_inputs_to_str(value: Any) -> str:
    """Parse the inputs to a string compatible with the judges API"""
    if is_none_or_nan(value):
        # The DBX managed backend doesn't allow empty inputs. This is
        # a temporary workaround to bypass the validation.
        return " "
    if isinstance(value, str):
        return value

    value = _to_dict(value)

    # Handle case where _to_dict returns a non-dict (e.g., a list that gets serialized
    # and remains a list)
    if not isinstance(value, dict):
        return json.dumps(value, cls=TraceJSONEncoder)

    if (messages := value.get(_MESSAGES_KEY)) and len(messages) > 0:
        contents = [m.get(_CONTENT_KEY) for m in messages]
        if len(contents) > 1 and all(isinstance(c, str) for c in contents):
            return json.dumps(messages)
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
    Extracts all top-level retrieval spans from the trace if there are multiple retrieval spans.
    If the trace does not have a retrieval span, return an empty dictionary.
    ⚠️ Warning: Please make sure to not throw exception. If fails, return an empty dictionary.
    """
    if trace is None or trace.data is None:
        return {}

    top_level_retrieval_spans = _get_top_level_retrieval_spans(trace)
    if len(top_level_retrieval_spans) == 0:
        return {}

    retrieved = {}

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
            top_level_retrieval_spans.append(span)

    return top_level_retrieval_spans


def _parse_chunk(chunk: Any) -> dict[str, Any] | None:
    if not isinstance(chunk, dict):
        return None

    doc = {"content": chunk.get("page_content")}
    if doc_uri := chunk.get("metadata", {}).get("doc_uri"):
        doc["doc_uri"] = doc_uri
    return doc


def clean_up_extra_traces(traces: list[Trace], eval_start_time: int) -> list[Trace]:
    """
    Clean up noisy traces generated outside predict function.

    Evaluation run should only contain traces that is being evaluated or generated by the predict
    function. If not, the result will not show the correct list of traces.
    Sometimes, there are extra traces generated during the evaluation, for example, custom scorer
    code might generate traces. This function cleans up those noisy traces.

    Args:
        traces: List of traces to clean up.
        eval_start_time: The start time of the evaluation run.

    Returns:
        List of traces that are kept after cleaning up extra traces.
    """
    from mlflow.tracking.fluent import _get_experiment_id

    try:
        extra_trace_ids = [
            trace.info.trace_id
            for trace in traces
            if not _should_keep_trace(trace, eval_start_time)
        ]
        if extra_trace_ids:
            _logger.debug(
                f"Found {len(extra_trace_ids)} extra traces generated during evaluation run. "
                "Deleting them."
            )
            # Import MlflowClient locally to avoid issues with tracing-only SDK
            from mlflow.tracking.client import MlflowClient

            MlflowClient().delete_traces(
                experiment_id=_get_experiment_id(), trace_ids=extra_trace_ids
            )
            for trace_id in extra_trace_ids:
                IPythonTraceDisplayHandler.get_instance().traces_to_display.pop(trace_id, None)
        else:
            _logger.debug("No extra traces found during evaluation run.")
    except Exception as e:
        _logger.debug(
            f"Failed to clean up extra traces generated during evaluation. The "
            f"result page might not show the correct list of traces. Error: {e}"
        )


def _should_keep_trace(trace: Trace, eval_start_time: int) -> bool:
    # We should not delete traces that are generated before the evaluation run started.
    if trace.info.timestamp_ms < eval_start_time:
        return True

    # If the scorer tracing is enabled, keep traces generated by scorers.
    if (
        MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING.get()
        and TraceTagKey.SOURCE_SCORER_NAME in trace.info.tags
    ):
        return True

    # Otherwise, only keep traces from the prediction function.
    return TraceTagKey.EVAL_REQUEST_ID in trace.info.tags


def construct_eval_result_df(
    run_id: str,
    traces: list[Trace],
    eval_results: list["EvalResult"],
) -> "pd.DataFrame | None":
    """
    Construct a pandas DataFrame from the traces and eval results.

    Args:
        run_id: The MLflow run ID of the evaluation run.
        traces: List of traces. Only TraceInfo is used here, and **spans are ignored&**.
            The expected input to this function is the result of
            `mlflow.search_traces(include_spans=False, return_type="list")`.
        eval_results: List of eval results containing the full spans.

    Returns:
        A pandas DataFrame with the eval results.
    """
    import pandas as pd

    if not traces:
        return None

    try:
        trace_id_to_info = {t.info.trace_id: t.info for t in traces}
        traces = [
            Trace(
                info=trace_id_to_info[eval_result.eval_item.trace.info.trace_id],
                data=eval_result.eval_item.trace.data,
            )
            for eval_result in eval_results
        ]
        df = traces_to_df(traces)
        # Add unpacked assessment columns. The result df should look like:
        # [trace_id, score_1/value, score_2/value, trace, state, ...]
        assessments = (
            df["assessments"].apply(lambda x: _get_assessment_values(x, run_id)).apply(pd.Series)
        )
        trace_id_column = df.pop("trace_id")
        return pd.concat([trace_id_column, assessments, df], axis=1)
    except Exception as e:
        _logger.debug(f"Failed to construct eval result DataFrame: {e}", exc_info=True)


def _get_assessment_values(assessments: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    result = {}
    for a in assessments:
        if (
            # Exclude feedbacks from other evaluation runs
            (source_run_id := a.get("metadata", {}).get(AssessmentMetadataKey.SOURCE_RUN_ID))
            and source_run_id != run_id
        ):
            continue
        if feedback := a.get("feedback"):
            result[f"{a['assessment_name']}/value"] = feedback.get("value")
        elif expectation := a.get("expectation"):
            result[f"{a['assessment_name']}/value"] = expectation.get("value")

    return result


def create_minimal_trace(eval_item: "EvalItem") -> Trace:
    """
    Create a minimal trace object with a single span, based on given inputs/outputs.

    If the eval_item has a source with session metadata (from a dataset created from traces),
    the session metadata will be restored on the newly created trace. This enables session-level
    scorers to identify which traces belong to the same session.
    """
    from mlflow.pyfunc.context import Context, set_prediction_context

    # Extract session metadata from source if available
    session_metadata = {}
    if eval_item.source and hasattr(eval_item.source, "source_data"):
        source_data = eval_item.source.source_data
        if session_id := source_data.get("session_id"):
            session_metadata[TraceMetadataKey.TRACE_SESSION] = session_id

    context = Context(request_id=eval_item.request_id, is_evaluate=True)
    with set_prediction_context(context):
        with mlflow.start_span(name="root_span", span_type=SpanType.CHAIN) as root_span:
            root_span.set_inputs(eval_item.inputs)
            root_span.set_outputs(eval_item.outputs)

            # Set session metadata on the trace while it's still active
            if session_metadata:
                mlflow.update_current_trace(metadata=session_metadata)

        return mlflow.get_trace(root_span.trace_id)


# MB: Caching on tracking URI level to avoid unnecessary checks for each trace.
@cached(cache={}, key=lambda **kwargs: kwargs["tracking_uri"])
def _does_store_support_trace_linking(*, tracking_uri: str, trace: Trace, run_id: str) -> bool:
    # Databricks backend is guaranteed to support trace linking
    if is_databricks_uri(tracking_uri):
        return True

    try:
        MlflowClient(tracking_uri).link_traces_to_run([trace.info.trace_id], run_id=run_id)
        return True
    except Exception:
        return False


def batch_link_traces_to_run(
    run_id: str | None, eval_results: list["EvalResult"], max_batch_size: int = 100
) -> None:
    """
    Batch link traces to a run to avoid rate limits.

    Args:
        run_id: The MLflow run ID to link traces to
        eval_results: List of evaluation results containing traces
        max_batch_size: Maximum number of traces to link per batch call
    """
    trace_ids = [eval_result.eval_item.trace.info.trace_id for eval_result in eval_results]
    # Batch the trace IDs to avoid overwhelming the MLflow backend
    for i in range(0, len(trace_ids), max_batch_size):
        batch = trace_ids[i : i + max_batch_size]
        try:
            MlflowClient().link_traces_to_run(run_id=run_id, trace_ids=batch)
        except Exception as e:
            # FileStore doesn't support trace linking, so we skip it
            if "Linking traces to runs is not supported in FileStore." in str(e):
                return

            _logger.warning(f"Failed to link batch of traces to run: {e}")


class ExtractedToolsFromTrace(BaseModel):
    tools: list["ChatTool"] = Field(
        default_factory=list,
        description="List of all available tools found in the trace",
    )

    model_config = {"extra": "forbid"}


def extract_available_tools_from_trace(trace: Trace, model: str | None = None) -> list["ChatTool"]:
    """
    Extract available tools from a trace by checking all LLM spans.

    This function uses a two-stage approach:
    1. Programmatic extraction: Checks all LLM and CHAT_MODEL spans for tools in
       attributes (mlflow.chat.tools) and inputs (inputs.tools field).
    2. LLM fallback: If no tools are found programmatically, uses an LLM to analyze
       the trace and identify tool definitions.

    The programmatic approach mirrors the frontend's getChatToolsFromSpan logic in
    ModelTraceExplorer.utils.tsx, which extracts tools per-span and returns a
    deduplicated list of all unique tools found across the trace.

    Args:
        trace: MLflow trace object
        model: Optional model URI to use for LLM-based fallback extraction
               (e.g., "openai:/gpt-4"). If None, uses a default model.

    Returns:
        List of unique ChatTool objects, or an empty list if no valid tools are found.
    """
    # Stage 1: Programmatic extraction from span attributes and inputs
    all_tools = []
    seen_tool_signatures = set()

    relevant_span_types = [SpanType.LLM, SpanType.CHAT_MODEL]

    for span in trace.data.spans:
        span_type = span.get_attribute(SpanAttributeKey.SPAN_TYPE)
        if span_type not in relevant_span_types:
            continue

        span_tools = _extract_tools_from_span(span)

        for tool in span_tools:
            if tool.function:
                tool_signature = _get_tool_signature(tool)
                if tool_signature not in seen_tool_signatures:
                    seen_tool_signatures.add(tool_signature)
                    all_tools.append(tool)

    if all_tools:
        return all_tools

    # Stage 2: LLM fallback when programmatic extraction yields no results
    return _try_extract_available_tools_with_llm(trace, model)


def _get_tool_signature(tool: "ChatTool") -> str:
    if not tool.function:
        return ""

    try:
        tool_dict = tool.function.model_dump()
    except AttributeError:
        tool_dict = tool.function.dict()

    return json.dumps(tool_dict, sort_keys=True)


def _extract_tools_from_span(span: Span) -> list["ChatTool"]:
    """
    Extract tools from a single LLM or CHAT_MODEL span, checking attribute first, then inputs.

    This mirrors the frontend's getChatToolsFromSpan logic exactly, but returns
    validated ChatTool objects using Pydantic validation.

    Args:
        span: MLflow span object

    Returns:
        List of ChatTool objects for this span
    """
    tools_attribute = span.get_attribute(SpanAttributeKey.CHAT_TOOLS)
    if tools_attribute is not None:
        try:
            if isinstance(tools_attribute, str):
                tools_attribute = json.loads(tools_attribute)
            return _parse_tools_to_chat_tool(tools_attribute)
        except Exception as e:
            _logger.debug(f"Failed to parse tools from attribute in span {span.span_id}: {e}")

    if span.inputs is not None:
        try:
            inputs = _to_dict(span.inputs)
            if "tools" in inputs:
                return _parse_tools_to_chat_tool(inputs["tools"])
        except Exception as e:
            _logger.debug(f"Failed to parse tools from inputs in span {span.span_id}: {e}")

    return []


def _parse_tools_to_chat_tool(tools_data: list[dict[str, Any]]) -> list["ChatTool"]:
    """
    Parse a list of tool dictionaries into ChatTool objects using Pydantic validation.

    Args:
        tools_data: List of tool dictionaries

    Returns:
        List of validated ChatTool objects. Invalid tools are skipped with debug logging.
    """
    from mlflow.types.chat import ChatTool

    validated_tools = []
    for data in tools_data:
        try:
            tool = ChatTool(**data)
            validated_tools.append(tool)
        except Exception as e:
            _logger.debug(f"Skipping invalid tool {data}: {e}")

    return validated_tools


def _try_extract_available_tools_with_llm(
    trace: Trace, model: str | None = None
) -> list["ChatTool"]:
    """
    Attempt to extract available tools from trace using LLM with structured output.

    This is a fallback method when programmatic extraction fails. It uses an LLM to
    analyze the trace and identify tool definitions that were available to the agent.

    Args:
        trace: MLflow trace object to analyze
        model: Optional model URI to use for extraction (e.g., "openai:/gpt-4").
               If None, uses a default model.

    Returns:
        List of ChatTool objects extracted by the LLM, or empty list if extraction fails.
    """
    if model is None:
        if is_databricks_uri(mlflow.get_tracking_uri()):
            # TODO: Add support for Databricks tool extraction with LLM fallback.
            _logger.warning("Databricks is not supported for tool extraction with LLM fallback.")
            return []
        else:
            model = "openai:/gpt-4.1-mini"

    try:
        from mlflow.types.chat import (
            ChatTool,
            FunctionParams,
            FunctionToolDefinition,
            ParamProperty,
        )

        output_example = json.dumps(
            ExtractedToolsFromTrace(
                tools=[
                    ChatTool(
                        type="function",
                        function=FunctionToolDefinition(
                            name="example_tool",
                            description="Description of what the tool does",
                            parameters=FunctionParams(
                                type="object",
                                properties={
                                    "param1": ParamProperty(
                                        type="string",
                                        description="A parameter",
                                    )
                                },
                                required=["param1"],
                            ),
                        ),
                    )
                ]
            ).model_dump(),
            indent=2,
        )

        messages = get_available_tools_extraction_prompts(output_example)

        result = get_chat_completions_with_structured_output(
            model_uri=model,
            messages=messages,
            output_schema=ExtractedToolsFromTrace,
            trace=trace,
        )

        return result.tools

    except Exception as e:
        _logger.warning(
            f"Failed to extract tools from trace using LLM. Returning empty list. Error: {e!r}"
        )
        return []
