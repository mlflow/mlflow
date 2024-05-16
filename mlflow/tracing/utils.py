from __future__ import annotations

import inspect
import json
import logging
import uuid
from collections import Counter, defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Literal, NamedTuple, Optional, Union

from opentelemetry import trace as trace_api
from packaging.version import Version

from mlflow.exceptions import BAD_REQUEST, MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.mlflow_tags import IMMUTABLE_TAGS

_logger = logging.getLogger(__name__)

SPANS_COLUMN_NAME = "spans"

if TYPE_CHECKING:
    import pandas

    import mlflow.entities
    from mlflow.entities import LiveSpan, Trace


def capture_function_input_args(func, args, kwargs) -> Dict[str, Any]:
    try:
        # Avoid capturing `self`
        func_signature = inspect.signature(func)
        bound_arguments = func_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        # Remove `self` from bound arguments if it exists
        if bound_arguments.arguments.get("self"):
            del bound_arguments.arguments["self"]

        return bound_arguments.arguments
    except Exception:
        _logger.warning(f"Failed to capture inputs for function {func.__name__}.")
        return {}


class TraceJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for serializing non-OpenTelemetry compatible objects in a trace or span.

    Trace may contain types that require custom serialization logic, such as Pydantic models,
    non-JSON-serializable types, etc.
    """

    def default(self, obj):
        try:
            # LangChain does some trick to keep both Pydantic 1.x and 2.x support, so checking
            # type with installed Pydantic version might not work for some models.
            # https://github.com/langchain-ai/langchain/blob/b66a4f48fa5656871c3e849f7e1790dfb5a4c56b/libs/core/langchain_core/pydantic_v1/__init__.py#L7
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

        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


@lru_cache(maxsize=1)
def encode_span_id(span_id: int) -> str:
    """
    Encode the given integer span ID to a 16-byte hex string.
    # https://github.com/open-telemetry/opentelemetry-python/blob/9398f26ecad09e02ad044859334cd4c75299c3cd/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py#L507-L508
    """
    return f"0x{trace_api.format_span_id(span_id)}"


@lru_cache(maxsize=1)
def encode_trace_id(trace_id: int) -> str:
    """
    Encode the given integer trace ID to a 32-byte hex string.
    """
    return f"0x{trace_api.format_trace_id(trace_id)}"


def decode_id(span_or_trace_id: str) -> int:
    """
    Decode the given hex string span or trace ID to an integer.
    """
    return int(span_or_trace_id, 16)


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


def deduplicate_span_names_in_place(spans: List[LiveSpan]):
    """
    Deduplicate span names in the trace data by appending an index number to the span name.

    This is only applied when there are multiple spans with the same name. The span names
    are modified in place to avoid unnecessary copying.

    E.g.
        ["red", "red"] -> ["red_1", "red_2"]
        ["red", "red", "blue"] -> ["red_1", "red_2", "blue"]

    Args:
        trace_data: The trace data object to deduplicate span names.
    """
    span_name_counter = Counter(span.name for span in spans)
    # Apply renaming only for duplicated spans
    span_name_counter = {name: 1 for name, count in span_name_counter.items() if count > 1}
    # Add index to the duplicated span names
    for span in spans:
        if count := span_name_counter.get(span.name):
            span_name_counter[span.name] += 1
            span._span._name = f"{span.name}_{count}"


def get_otel_attribute(span: trace_api.Span, key: str) -> Optional[str]:
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
        return json.loads(span.attributes.get(key))
    except Exception:
        _logger.debug(f"Failed to get attribute {key} with from span {span}.", exc_info=True)


def maybe_get_request_id(is_evaluate=False) -> Optional[str]:
    """Get the request ID if the current prediction is as a part of MLflow model evaluation."""
    # NB: Tracing is enabled in mlflow-skinny, but the pyfunc module cannot be imported as it
    #     relies on numpy, which is not installed in skinny.
    try:
        from mlflow.pyfunc.context import get_prediction_context
    except ImportError:
        return None

    context = get_prediction_context()
    if not context or (is_evaluate and not context.is_evaluate):
        return None

    if not context.request_id:
        raise MlflowException(
            f"Missing request_id for context {context}.",
            error_code=BAD_REQUEST,
        )

    return context.request_id


def traces_to_df(traces: List[Trace]) -> "pandas.DataFrame":
    """
    Convert a list of MLflow Traces to a pandas DataFrame with one column called "traces"
    containing string representations of each Trace.
    """
    import pandas as pd

    rows = [trace.to_pandas_dataframe_row() for trace in traces]
    return pd.DataFrame.from_records(rows)


def extract_span_inputs_outputs(
    traces: Union[List["mlflow.entities.Trace"], "pandas.DataFrame"],
    fields: List[str],
    col_name: Optional[str] = None,
) -> "pandas.DataFrame":
    """
    Extracts the specified input and output fields from the spans contained in the specified traces.

    Args:
        traces: A list of :py:class:`mlflow.entities.Trace` or a pandas DataFrame containing traces.
        fields: A list of field strings of the form 'span_name.[inputs|outputs]' or
            'span_name.[inputs|outputs].field_name'.
        col_name: The name of the column in the traces DataFrame containing the spans. If `traces`
            is a list of MLflow Traces, this argument should not be provided.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise MlflowException(
            message=(
                "The `pandas` library is not installed. Please install `pandas` to use the"
                f"`mlflow.tracing.extract` function. Error: {e}"
            ),
        )

    parsed_fields = _parse_fields(fields)

    if isinstance(traces, list):
        if col_name is not None:
            raise MlflowException(
                message=(
                    "If `traces` is a list of MLflow Traces, `col_name` should not be provided."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        traces = traces_to_df(traces)
        col_name = SPANS_COLUMN_NAME

    if isinstance(traces, pd.DataFrame):
        return _extract_from_traces_pandas_df(df=traces, col_name=col_name, fields=parsed_fields)

    raise MlflowException(
        message=(
            "`traces` must be a list of MLflow Traces or a pandas DataFrame. Got: {type(traces)}"
        ),
        error_code=INVALID_PARAMETER_VALUE,
    )


class _ParsedField(NamedTuple):
    """
    Represents a parsed field from a string of the form 'span_name.[inputs|outputs]' or
    'span_name.[inputs|outputs].field_name'.
    """

    span_name: str
    field_type: Literal["inputs", "outputs"]
    field_name: Optional[str]

    @classmethod
    def from_string(cls, s: str) -> "_ParsedField":
        components = s.split(".")
        if len(components) not in [2, 3] or components[1] not in ["inputs", "outputs"]:
            raise MlflowException(
                message=(
                    f"Field must be of the form 'span_name.[inputs|outputs]' or"
                    f" 'span_name.[inputs|outputs].field_name'. Got: {s}"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

        return cls(
            span_name=components[0],
            field_type=components[1],
            field_name=components[2] if len(components) == 3 else None,
        )

    def __str__(self) -> str:
        return (
            f"{self.span_name}.{self.field_type}.{self.field_name}"
            if self.field_name is not None
            else f"{self.span_name}.{self.field_type}"
        )


def _parse_fields(fields: List[str]) -> List["_ParsedField"]:
    """
    Parses the specified field strings of the form 'span_name.[inputs|outputs]' or
    'span_name.[inputs|outputs].field_name' into _ParsedField objects.
    """
    return [_ParsedField.from_string(field) for field in fields]


def _extract_from_traces_pandas_df(
    df: "pandas.DataFrame", col_name: str, fields: List["_ParsedField"]
) -> "pandas.DataFrame":
    """
    Extracts the specified fields from the spans contained in the specified column of the
    specified traces DataFrame.
    """

    from mlflow.entities import Span

    if col_name not in df.columns:
        raise MlflowException(
            message=(
                f"Column '{col_name}' not found in traces DataFrame."
                f" Available columns: {df.columns}"
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    new_columns: Dict[str, List[Any]] = defaultdict(list)
    for _, row in df.iterrows():
        spans_dict: Dict[str, List[Span]] = defaultdict(list)
        for span in _extract_spans_from_row(row[col_name]):
            spans_dict[span.name].append(span)

        for field in fields:
            matching_spans = spans_dict.get(field.span_name, [])
            matching_value = _find_matching_value(field, matching_spans)
            new_columns[str(field)].append(matching_value)

    df_with_new_fields = df.copy()
    for field in fields:
        df_with_new_fields[str(field)] = new_columns[str(field)]

    return df_with_new_fields


def _find_matching_value(
    field: "_ParsedField", spans: List["mlflow.entities.Span"]
) -> Optional[Any]:
    """
    Find the value of the field in the list of spans. If the field is not found, return None.
    """
    for span in spans:
        span_inputs_or_outputs = getattr(span, field.field_type)
        if (
            isinstance(span_inputs_or_outputs, dict)
            and field.field_name is not None
            and field.field_name in span_inputs_or_outputs
        ):
            return span_inputs_or_outputs.get(field.field_name)
        elif field.field_name is None:
            return span_inputs_or_outputs


def _extract_spans_from_row(
    row_content: Optional[List[Dict[str, Any]]],
) -> List["mlflow.entities.Span"]:
    """
    Parses and extracts MLflow Spans from the row content of a traces pandas DataFrame.
    """
    from mlflow.entities import Span

    if row_content is None:
        return []

    try:
        return [Span.from_dict(span_dict) for span_dict in row_content]
    except Exception as e:
        raise MlflowException(
            message=(
                f"Failed to extract spans from traces DataFrame row content: {row_content}."
                f" Error: {e}"
            ),
            error_code=INVALID_PARAMETER_VALUE,
        ) from e


def exclude_immutable_tags(tags: Dict[str, str]) -> Dict[str, str]:
    """Exclude immutable tags e.g. "mlflow.user" from the given tags."""
    return {k: v for k, v in tags.items() if k not in IMMUTABLE_TAGS}


def generate_request_id() -> str:
    return uuid.uuid4().hex
