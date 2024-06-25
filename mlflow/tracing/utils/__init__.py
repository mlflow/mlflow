# TODO: Split this file into multiple files and move under utils directory.
from __future__ import annotations

import inspect
import json
import logging
import uuid
from collections import Counter
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opentelemetry import trace as trace_api
from packaging.version import Version

from mlflow.exceptions import BAD_REQUEST, MlflowTracingException
from mlflow.utils.mlflow_tags import IMMUTABLE_TAGS

_logger = logging.getLogger(__name__)

SPANS_COLUMN_NAME = "spans"

if TYPE_CHECKING:
    from mlflow.entities import LiveSpan


def capture_function_input_args(func, args, kwargs) -> Dict[str, Any]:
    # Avoid capturing `self`
    func_signature = inspect.signature(func)
    bound_arguments = func_signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    # Remove `self` from bound arguments if it exists
    if bound_arguments.arguments.get("self"):
        del bound_arguments.arguments["self"]

    return bound_arguments.arguments


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
        spans: A list of spans to deduplicate.
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


def _try_get_prediction_context():
    # NB: Tracing is enabled in mlflow-skinny, but the pyfunc module cannot be imported as it
    #     relies on numpy, which is not installed in skinny.
    try:
        from mlflow.pyfunc.context import get_prediction_context
    except ImportError:
        return

    return get_prediction_context()


def maybe_get_request_id(is_evaluate=False) -> Optional[str]:
    """Get the request ID if the current prediction is as a part of MLflow model evaluation."""
    context = _try_get_prediction_context()
    if not context or (is_evaluate and not context.is_evaluate):
        return None

    if not context.request_id and is_evaluate:
        raise MlflowTracingException(
            f"Missing request_id for context {context}. "
            "request_id can't be None when is_evaluate=True.",
            error_code=BAD_REQUEST,
        )

    return context.request_id


def maybe_get_dependencies_schemas() -> Optional[dict]:
    context = _try_get_prediction_context()
    if context:
        return context.dependencies_schemas


def exclude_immutable_tags(tags: Dict[str, str]) -> Dict[str, str]:
    """Exclude immutable tags e.g. "mlflow.user" from the given tags."""
    return {k: v for k, v in tags.items() if k not in IMMUTABLE_TAGS}


def generate_request_id() -> str:
    return uuid.uuid4().hex
