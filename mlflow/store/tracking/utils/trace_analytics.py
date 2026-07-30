import json
import math
from typing import Any

from mlflow.tracing.constant import CostKey, TokenUsageKey, TraceMetadataKey

MODEL_DIMENSION_MAX_LENGTH = 500

TOKEN_COLUMN_BY_KEY = {
    TokenUsageKey.INPUT_TOKENS: "input_tokens",
    TokenUsageKey.OUTPUT_TOKENS: "output_tokens",
    TokenUsageKey.TOTAL_TOKENS: "total_tokens",
    TokenUsageKey.CACHE_READ_INPUT_TOKENS: "cache_read_input_tokens",
    TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: "cache_creation_input_tokens",
}
COST_COLUMN_BY_KEY = {
    CostKey.INPUT_COST: "input_cost",
    CostKey.OUTPUT_COST: "output_cost",
    CostKey.TOTAL_COST: "total_cost",
}


def finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def analytics_columns_from_metadata(metadata: dict[str, str]) -> dict[str, float | str | None]:
    columns = {}
    if TraceMetadataKey.TRACE_SESSION in metadata:
        columns["session_id"] = metadata[TraceMetadataKey.TRACE_SESSION]
    if TraceMetadataKey.TOKEN_USAGE in metadata:
        token_usage = _json_object(metadata[TraceMetadataKey.TOKEN_USAGE])
        columns.update({
            column: finite_float_or_none(token_usage.get(key))
            for key, column in TOKEN_COLUMN_BY_KEY.items()
        })
    if TraceMetadataKey.COST in metadata:
        cost = _json_object(metadata[TraceMetadataKey.COST])
        columns.update({
            column: finite_float_or_none(cost.get(key))
            for key, column in COST_COLUMN_BY_KEY.items()
        })
    return columns


def compatibility_metadata_from_columns(sql_trace_info) -> dict[str, str]:
    metadata = {}
    if sql_trace_info.session_id is not None:
        metadata[TraceMetadataKey.TRACE_SESSION] = sql_trace_info.session_id

    token_usage = {
        key: value
        for key, column in TOKEN_COLUMN_BY_KEY.items()
        if (value := getattr(sql_trace_info, column)) is not None
    }
    if token_usage:
        metadata[TraceMetadataKey.TOKEN_USAGE] = json.dumps(token_usage)

    cost = {
        key: value
        for key, column in COST_COLUMN_BY_KEY.items()
        if (value := getattr(sql_trace_info, column)) is not None
    }
    if cost:
        metadata[TraceMetadataKey.COST] = json.dumps(cost)
    return metadata


def assessment_aggregate(value: Any) -> tuple[float | None, bool]:
    if isinstance(value, bool):
        return (1.0 if value else 0.0), False
    if isinstance(value, (int, float)):
        value = float(value)
        return (value, True) if math.isfinite(value) else (None, False)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "no"}:
            return (1.0 if normalized == "yes" else 0.0), False
    return None, False


def bounded_model_dimension(value: Any) -> str | None:
    return value[:MODEL_DIMENSION_MAX_LENGTH] if isinstance(value, str) else None
