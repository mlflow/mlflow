import inspect
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opentelemetry import trace as trace_api
from packaging.version import Version

if TYPE_CHECKING:
    import pandas

    import mlflow.entities

_logger = logging.getLogger(__name__)

SPANS_COLUMN_NAME = "spans"


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


def traces_to_df(traces: List["mlflow.entities.Trace"]) -> "pandas.DataFrame":
    """
    Convert a list of MLflow Traces to a pandas DataFrame with one column called "traces"
    containing string representations of each Trace.
    """
    import pandas as pd

    rows = [
        _TraceRow(
            request_id=trace.info.request_id,
            timestamp_ms=trace.info.timestamp_ms,
            status=trace.info.status,
            execution_time_ms=trace.info.execution_time_ms,
            request=trace.data.request,
            request_metadata=trace.info.request_metadata,
            spans=trace.data.spans,
            tags=trace.info.tags,
            response=trace.data.response,
        )
        for trace in traces
    ]
    return pd.DataFrame.from_records([row.to_dict() for row in rows])


@dataclass
class _TraceRow:
    request_id: str
    timestamp_ms: int
    status: str
    execution_time_ms: int
    request: str
    request_metadata: Dict[str, str]
    spans: List["mlflow.entities.Span"]
    tags: Dict[str, str]
    response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp_ms": self.timestamp_ms,
            "status": self.status,
            "execution_time_ms": self.execution_time_ms,
            "request": self.request,
            "response": self.response,
            "request_metadata": self.request_metadata,
            "spans": [span.to_dict() for span in self.spans],
            "tags": self.tags,
        }
