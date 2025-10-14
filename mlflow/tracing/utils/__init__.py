# TODO: Split this file into multiple files and move under utils directory.
from __future__ import annotations

import inspect
import json
import logging
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Generator

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from packaging.version import Version

from mlflow.exceptions import BAD_REQUEST, MlflowException, MlflowTracingException
from mlflow.tracing.constant import (
    ASSESSMENT_ID_PREFIX,
    TRACE_ID_V4_PREFIX,
    TRACE_REQUEST_ID_PREFIX,
    SpanAttributeKey,
    TokenUsageKey,
    TraceMetadataKey,
    TraceSizeStatsKey,
)
from mlflow.utils.mlflow_tags import IMMUTABLE_TAGS
from mlflow.version import IS_TRACING_SDK_ONLY

_logger = logging.getLogger(__name__)

SPANS_COLUMN_NAME = "spans"

if TYPE_CHECKING:
    from mlflow.entities import LiveSpan, Trace
    from mlflow.pyfunc.context import Context
    from mlflow.types.chat import ChatTool


def capture_function_input_args(func, args, kwargs) -> dict[str, Any] | None:
    try:
        func_signature = inspect.signature(func)
        bound_arguments = func_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        # Remove `self` from bound arguments if it exists
        if bound_arguments.arguments.get("self"):
            del bound_arguments.arguments["self"]

        # Remove `cls` from bound arguments if it's the first parameter and it's a type
        # This detects classmethods more reliably
        params = list(bound_arguments.arguments.keys())
        if params and params[0] == "cls" and isinstance(bound_arguments.arguments["cls"], type):
            del bound_arguments.arguments["cls"]

        return bound_arguments.arguments
    except Exception:
        _logger.warning(f"Failed to capture inputs for function {func.__name__}.")
        return None


class TraceJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for serializing non-OpenTelemetry compatible objects in a trace or span.

    Trace may contain types that require custom serialization logic, such as Pydantic models,
    non-JSON-serializable types, etc.
    """

    def default(self, obj):
        try:
            import langchain

            # LangChain < 0.3.0 does some trick to support Pydantic 1.x and 2.x, so checking
            # type with installed Pydantic version might not work for some models.
            # https://github.com/langchain-ai/langchain/blob/b66a4f48fa5656871c3e849f7e1790dfb5a4c56b/libs/core/langchain_core/pydantic_v1/__init__.py#L7
            if Version(langchain.__version__) < Version("0.3.0"):
                from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel

                if isinstance(obj, LangChainBaseModel):
                    return obj.dict()
        except ImportError:
            pass

        try:
            import pydantic

            if isinstance(obj, pydantic.BaseModel):
                # NB: Pydantic 2.0+ has a different API for model serialization
                if Version(pydantic.VERSION) >= Version("2.0"):
                    return obj.model_dump()
                else:
                    return obj.dict()
        except ImportError:
            pass

        # Some dataclass object defines __str__ method that doesn't return the full object
        # representation, so we use dict representation instead.
        # E.g. https://github.com/run-llama/llama_index/blob/29ece9b058f6b9a1cf29bc723ed4aa3a39879ad5/llama-index-core/llama_index/core/chat_engine/types.py#L63-L64
        if is_dataclass(obj):
            try:
                return asdict(obj)
            except TypeError:
                pass

        # Some object has dangerous side effect in __str__ method, so we use class name instead.
        if not self._is_safe_to_encode_str(obj):
            return type(obj)

        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

    def _is_safe_to_encode_str(self, obj) -> bool:
        """Check if it's safe to encode the object as a string."""
        try:
            # These Llama Index objects are not safe to encode as string, because their __str__
            # method consumes the stream and make it unusable.
            # E.g. https://github.com/run-llama/llama_index/blob/54f2da61ba8a573284ab8336f2b2810d948c3877/llama-index-core/llama_index/core/base/response/schema.py#L120-L127
            from llama_index.core.base.response.schema import (
                AsyncStreamingResponse,
                StreamingResponse,
            )
            from llama_index.core.chat_engine.types import StreamingAgentChatResponse

            if isinstance(
                obj,
                (AsyncStreamingResponse, StreamingResponse, StreamingAgentChatResponse),
            ):
                return False
        except ImportError:
            pass

        return True


@lru_cache(maxsize=1)
def encode_span_id(span_id: int) -> str:
    """
    Encode the given integer span ID to a 16-byte hex string.
    # https://github.com/open-telemetry/opentelemetry-python/blob/9398f26ecad09e02ad044859334cd4c75299c3cd/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py#L507-L508
    # NB: We don't add '0x' prefix to the hex string here for simpler parsing in backend.
    #   Some backend (e.g. Databricks) disallow this prefix.
    """
    return trace_api.format_span_id(span_id)


@lru_cache(maxsize=1)
def encode_trace_id(trace_id: int) -> str:
    """
    Encode the given integer trace ID to a 32-byte hex string.
    """
    return trace_api.format_trace_id(trace_id)


def decode_id(span_or_trace_id: str) -> int:
    """
    Decode the given hex string span or trace ID to an integer.
    """
    return int(span_or_trace_id, 16)


def get_mlflow_span_for_otel_span(span: OTelSpan) -> LiveSpan | None:
    """
    Get the active MLflow span for the given OpenTelemetry span.
    """
    from mlflow.tracing.trace_manager import InMemoryTraceManager

    trace_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
    mlflow_span_id = encode_span_id(span.get_span_context().span_id)
    return InMemoryTraceManager.get_instance().get_span_from_id(trace_id, mlflow_span_id)


def build_otel_context(trace_id: int, span_id: int) -> trace_api.SpanContext:
    """
    Build an OpenTelemetry SpanContext object from the given trace and span IDs.
    """
    return trace_api.SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        # NB: This flag is OpenTelemetry's concept to indicate whether the context is
        # propagated from remote parent or not. We don't support distributed tracing
        # yet so always set it to False.
        is_remote=False,
    )


def deduplicate_span_names_in_place(spans: list[LiveSpan]):
    """
    Deduplicate span names in the trace data by appending an index number to the span name.

    This is only applied when there are multiple spans with the same name. The span names
    are modified in place to avoid unnecessary copying.

    E.g.
        ["red", "red"] -> ["red_1", "red_2"]
        ["red", "red", "blue"] -> ["red_1", "red_2", "blue"]

    Args:
        spans: A list of spans to deduplicate.
    """
    # Use _original_name to handle incremental deduplication correctly
    span_name_counter = Counter(span._original_name for span in spans)
    # Apply renaming only for duplicated spans
    span_name_counter = {name: 1 for name, count in span_name_counter.items() if count > 1}
    # Add index to the duplicated span names
    for span in spans:
        if count := span_name_counter.get(span._original_name):
            span_name_counter[span._original_name] += 1
            span._span._name = f"{span._original_name}_{count}"


def aggregate_usage_from_spans(spans: list[LiveSpan]) -> dict[str, int] | None:
    """Aggregate token usage information from all spans in the trace."""
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    has_usage_data = False

    span_id_to_spans = {span.span_id: span for span in spans}
    children_map: defaultdict[str, list[LiveSpan]] = defaultdict(list)
    roots: list[LiveSpan] = []

    for span in spans:
        parent_id = span.parent_id
        if parent_id and parent_id in span_id_to_spans:
            children_map[parent_id].append(span)
        else:
            roots.append(span)

    def dfs(span: LiveSpan, ancestor_has_usage: bool) -> None:
        nonlocal input_tokens, output_tokens, total_tokens, has_usage_data

        usage = span.get_attribute(SpanAttributeKey.CHAT_USAGE)
        span_has_usage = usage is not None

        if span_has_usage and not ancestor_has_usage:
            input_tokens += usage.get(TokenUsageKey.INPUT_TOKENS, 0)
            output_tokens += usage.get(TokenUsageKey.OUTPUT_TOKENS, 0)
            total_tokens += usage.get(TokenUsageKey.TOTAL_TOKENS, 0)
            has_usage_data = True

        next_ancestor_has_usage = ancestor_has_usage or span_has_usage
        for child in children_map.get(span.span_id, []):
            dfs(child, next_ancestor_has_usage)

    for root in roots:
        dfs(root, False)

    # If none of the spans have token usage data, we shouldn't log token usage metadata.
    if not has_usage_data:
        return None

    return {
        TokenUsageKey.INPUT_TOKENS: input_tokens,
        TokenUsageKey.OUTPUT_TOKENS: output_tokens,
        TokenUsageKey.TOTAL_TOKENS: total_tokens,
    }


def get_otel_attribute(span: trace_api.Span, key: str) -> str | None:
    """
    Get the attribute value from the OpenTelemetry span in a decoded format.

    Args:
        span: The OpenTelemetry span object.
        key: The key of the attribute to retrieve.

    Returns:
        The attribute value as decoded string. If the attribute is not found or cannot
        be parsed, return None.
    """
    try:
        attribute_value = span.attributes.get(key)
        if attribute_value is None:
            return None
        return json.loads(attribute_value)
    except Exception:
        _logger.debug(f"Failed to get attribute {key} with from span {span}.", exc_info=True)


def _try_get_prediction_context():
    # NB: Tracing is enabled in mlflow-skinny, but the pyfunc module cannot be imported as it
    #     relies on numpy, which is not installed in skinny.
    try:
        from mlflow.pyfunc.context import get_prediction_context
    except (ImportError, KeyError):
        return

    return get_prediction_context()


def maybe_get_request_id(is_evaluate=False) -> str | None:
    """Get the request ID if the current prediction is as a part of MLflow model evaluation."""
    context = _try_get_prediction_context()
    if not context or (is_evaluate and not context.is_evaluate):
        return None

    if not context.request_id and is_evaluate:
        _logger.warning(
            f"Missing request_id for context {context}. request_id can't be None when "
            "is_evaluate=True. This is likely an internal error of MLflow, please file "
            "a bug report at https://github.com/mlflow/mlflow/issues."
        )
        return None

    return context.request_id


def maybe_get_dependencies_schemas() -> dict[str, Any] | None:
    context = _try_get_prediction_context()
    if context:
        return context.dependencies_schemas


def maybe_get_logged_model_id() -> str | None:
    """
    Get the logged model ID associated with the current prediction context.
    """
    if context := _try_get_prediction_context():
        return context.model_id


def exclude_immutable_tags(tags: dict[str, str]) -> dict[str, str]:
    """Exclude immutable tags e.g. "mlflow.user" from the given tags."""
    return {k: v for k, v in tags.items() if k not in IMMUTABLE_TAGS}


def generate_mlflow_trace_id_from_otel_trace_id(otel_trace_id: int) -> str:
    """
    Generate an MLflow trace ID from an OpenTelemetry trace ID.

    Args:
        otel_trace_id: The OpenTelemetry trace ID as an integer.

    Returns:
        The MLflow trace ID string in format "tr-<hex_trace_id>".
    """
    return TRACE_REQUEST_ID_PREFIX + encode_trace_id(otel_trace_id)


def generate_trace_id_v4_from_otel_trace_id(otel_trace_id: int, location: str) -> str:
    """
    Generate a trace ID in v4 format from the given OpenTelemetry trace ID.

    Args:
        otel_trace_id: The OpenTelemetry trace ID as an integer.
        location: The location, of the trace.

    Returns:
        The MLflow trace ID string in format "trace:/<location>/<hex_trace_id>".
    """
    return construct_trace_id_v4(location, encode_trace_id(otel_trace_id))


def generate_trace_id_v4(span: OTelSpan, location: str) -> str:
    """
    Generate a trace ID for the given span.

    Args:
        span: The OpenTelemetry span object.
        location: The location, of the trace.

    Returns:
        Trace ID with format "trace:/<location>/<hex_trace_id>".
    """
    return generate_trace_id_v4_from_otel_trace_id(span.context.trace_id, location)


def generate_trace_id_v3(span: OTelSpan) -> str:
    """
    Generate a trace ID for the given span (V3 trace schema).

    The format will be "tr-<trace_id>" where the trace_id is hex-encoded Otel trace ID.
    """
    return generate_mlflow_trace_id_from_otel_trace_id(span.context.trace_id)


def generate_request_id_v2() -> str:
    """
    Generate a request ID for the given span.

    This should only be used for V2 trace schema where we use a random UUID as
    request ID. In the V3 schema, "request_id" is renamed to "trace_id" and
    we use the otel-generated trace ID with encoding.
    """
    return uuid.uuid4().hex


def construct_full_inputs(func, *args, **kwargs) -> dict[str, Any]:
    """
    Construct the full input arguments dictionary for the given function,
    including positional and keyword arguments.
    """
    signature = inspect.signature(func)
    # this does not create copy. So values should not be mutated directly
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        arguments.pop("self")

    return arguments


@contextmanager
def maybe_set_prediction_context(context: "Context" | None):
    """
    Set the prediction context if the given context
    is not None. Otherwise no-op.
    """
    if not IS_TRACING_SDK_ONLY and context:
        from mlflow.pyfunc.context import set_prediction_context

        with set_prediction_context(context):
            yield
    else:
        yield


def set_span_chat_tools(span: LiveSpan, tools: list[ChatTool]):
    """
    Set the `mlflow.chat.tools` attribute on the specified span. This
    attribute is used in the UI, and also by downstream applications that
    consume trace data, such as MLflow evaluate.

    Args:
        span: The LiveSpan to add the attribute to
        tools: A list of standardized chat tool definitions (refer to the
              `spec <../llms/tracing/tracing-schema.html#chat-completion-spans>`_
              for details)

    Example:

    .. code-block:: python
        :test:

        import mlflow
        from mlflow.tracing import set_span_chat_tools

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]


        @mlflow.trace
        def f():
            span = mlflow.get_current_active_span()
            set_span_chat_tools(span, tools)
            return 0


        f()
    """
    from mlflow.types.chat import ChatTool

    if not isinstance(tools, list):
        raise MlflowTracingException(
            f"Invalid tools type {type(tools)}. Expected a list of ChatTool.",
            error_code=BAD_REQUEST,
        )

    sanitized_tools = []
    for tool in tools:
        if isinstance(tool, dict):
            ChatTool.validate_compat(tool)
            sanitized_tools.append(tool)
        elif isinstance(tool, ChatTool):
            sanitized_tools.append(tool.model_dump_compat(exclude_unset=True))

    span.set_attribute(SpanAttributeKey.CHAT_TOOLS, sanitized_tools)


def _calculate_percentile(sorted_data: list[float], percentile: float) -> float:
    """
    Calculate the percentile value from sorted data.

    Args:
        sorted_data: A sorted list of numeric values
        percentile: The percentile to calculate (e.g., 0.25 for 25th percentile)

    Returns:
        The percentile value
    """
    if not sorted_data:
        return 0.0

    n = len(sorted_data)
    index = percentile * (n - 1)
    lower = int(index)
    upper = lower + 1

    if upper >= n:
        return sorted_data[-1]

    # Linear interpolation between two nearest values
    weight = index - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


def add_size_stats_to_trace_metadata(trace: Trace):
    """
    Calculate the stats of trace and span sizes and add it as a metadata to the trace.

    This method modifies the trace object in place by adding a new tag.

    Note: For simplicity, we calculate the size without considering the size metadata itself.
    This provides a close approximation without requiring complex calculations.

    This function must not throw an exception.
    """
    from mlflow.entities import Trace, TraceData

    try:
        span_sizes = []
        for span in trace.data.spans:
            span_json = json.dumps(span.to_dict(), cls=TraceJSONEncoder)
            span_sizes.append(len(span_json.encode("utf-8")))

        # NB: To compute the size of the total trace, we need to include the size of the
        # the trace info and the parent dicts for the spans. To avoid serializing spans
        # again (which can be expensive), we compute the size of the trace without spans
        # and combine it with the total size of the spans.
        empty_trace = Trace(info=trace.info, data=TraceData(spans=[]))
        metadata_size = len((empty_trace.to_json()).encode("utf-8"))

        # NB: the third term is the size of comma separators between spans (", ").
        trace_size_bytes = sum(span_sizes) + metadata_size + (len(span_sizes) - 1) * 2

        # Sort span sizes for percentile calculation
        sorted_span_sizes = sorted(span_sizes)

        size_stats = {
            TraceSizeStatsKey.TOTAL_SIZE_BYTES: trace_size_bytes,
            TraceSizeStatsKey.NUM_SPANS: len(span_sizes),
            TraceSizeStatsKey.MAX_SPAN_SIZE_BYTES: max(span_sizes),
            TraceSizeStatsKey.P25_SPAN_SIZE_BYTES: int(
                _calculate_percentile(sorted_span_sizes, 0.25)
            ),
            TraceSizeStatsKey.P50_SPAN_SIZE_BYTES: int(
                _calculate_percentile(sorted_span_sizes, 0.50)
            ),
            TraceSizeStatsKey.P75_SPAN_SIZE_BYTES: int(
                _calculate_percentile(sorted_span_sizes, 0.75)
            ),
        }

        trace.info.trace_metadata[TraceMetadataKey.SIZE_STATS] = json.dumps(size_stats)
        # Keep the total size as a separate metadata for backward compatibility
        trace.info.trace_metadata[TraceMetadataKey.SIZE_BYTES] = str(trace_size_bytes)
    except Exception:
        _logger.warning("Failed to add size stats to trace metadata.", exc_info=True)


def update_trace_state_from_span_conditionally(trace, root_span):
    """
    Update trace state from span status, but only if the user hasn't explicitly set
    a different trace status.

    This utility preserves user-set trace status while maintaining default behavior
    for traces that haven't been explicitly configured. Used by trace processors when
    converting traces to an exportable state.

    Args:
        trace: The trace object to potentially update
        root_span: The root span whose status may be used to update the trace state
    """
    from mlflow.entities.trace_state import TraceState

    # Only update trace state from span status if trace is still IN_PROGRESS
    # If the trace state is anything else, it means the user explicitly set it
    # and we should preserve it
    if trace.info.state == TraceState.IN_PROGRESS:
        trace.info.state = TraceState.from_otel_status(root_span.status)


def get_experiment_id_for_trace(span: OTelReadableSpan) -> str:
    """
    Determine the experiment ID to associate with the trace.

    The experiment ID can be configured in multiple ways, in order of precedence:
      1. An experiment ID specified via the span creation API i.e. MlflowClient().start_trace()
      2. An experiment ID specified via `mlflow.tracing.set_destination`
      3. An experiment ID of an active run.
      4. The default experiment ID

    Args:
        span: The OpenTelemetry ReadableSpan to extract experiment ID from.

    Returns:
        The experiment ID string to use for the trace.
    """
    from mlflow.tracing.provider import _MLFLOW_TRACE_USER_DESTINATION
    from mlflow.tracking.fluent import _get_experiment_id, _get_latest_active_run

    if experiment_id := get_otel_attribute(span, SpanAttributeKey.EXPERIMENT_ID):
        return experiment_id

    if destination := _MLFLOW_TRACE_USER_DESTINATION.get():
        if exp_id := getattr(destination, "experiment_id", None):
            return exp_id

    if run := _get_latest_active_run():
        return run.info.experiment_id

    return _get_experiment_id()


def get_active_spans_table_name() -> str | None:
    """
    Get active Unity Catalog spans table name that's set by `mlflow.tracing.set_destination`.
    """
    from mlflow.entities.trace_location import UCSchemaLocation
    from mlflow.tracing.provider import _MLFLOW_TRACE_USER_DESTINATION

    if destination := _MLFLOW_TRACE_USER_DESTINATION.get():
        if isinstance(destination, UCSchemaLocation):
            return destination.full_otel_spans_table_name

    return None


def generate_assessment_id() -> str:
    """
    Generates an assessment ID of the form 'a-<uuid4>' in hex string format.

    Returns:
        A unique identifier for an assessment that will be logged to a trace tag.
    """
    id = uuid.uuid4().hex
    return f"{ASSESSMENT_ID_PREFIX}{id}"


@contextmanager
def _bypass_attribute_guard(span: OTelSpan) -> Generator[None, None, None]:
    """
    OpenTelemetry does not allow setting attributes if the span has end time defined.
    https://github.com/open-telemetry/opentelemetry-python/blob/d327927d0274a320466feec6fba6d6ddb287dc5a/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py#L849-L851

    However, we need to set some attributes within `on_end` handler of the span processor,
    where the span is already marked as ended. This context manager is a hacky workaround
    to bypass the attribute guard.
    """
    original_end_time = span._end_time
    span._end_time = None
    try:
        yield
    finally:
        span._end_time = original_end_time


def parse_trace_id_v4(trace_id: str | None) -> tuple[str | None, str | None]:
    """
    Parse the trace ID into location and trace ID components.
    """
    if trace_id is None:
        return None, None
    if trace_id.startswith(TRACE_ID_V4_PREFIX):
        match trace_id.removeprefix(TRACE_ID_V4_PREFIX).split("/"):
            case [location, tid] if location and tid:
                return location, tid
            case _:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid trace ID format: {trace_id}. "
                    f"Expected format: {TRACE_ID_V4_PREFIX}<location>/<trace_id>"
                )
    return None, trace_id


def construct_trace_id_v4(location: str, trace_id: str) -> str:
    """
    Construct a trace ID for the given location and trace ID.
    """
    return f"{TRACE_ID_V4_PREFIX}{location}/{trace_id}"
