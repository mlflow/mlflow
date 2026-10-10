import json
import math
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import and_, false, or_
from sqlalchemy.sql.elements import ColumnElement

from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import (
    MAX_CHARS_IN_TRACE_INFO_METADATA,
    MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE,
    CostKey,
    TokenUsageKey,
    TraceMetadataKey,
)
from mlflow.utils.validation import _validate_length_limit

MODEL_DIMENSION_MAX_LENGTH = 500
_BIGINT_MIN = -(2**63)
_BIGINT_MAX = 2**63 - 1

TOKEN_COLUMN_BY_KEY = {
    TokenUsageKey.INPUT_TOKENS: "input_tokens",
    TokenUsageKey.OUTPUT_TOKENS: "output_tokens",
    TokenUsageKey.TOTAL_TOKENS: "total_tokens",
    TokenUsageKey.CACHE_READ_INPUT_TOKENS: "cache_read_input_tokens",
    TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: "cache_creation_input_tokens",
    TokenUsageKey.CACHE_CREATION_INPUT_TOKENS_ABOVE_1HR: ("cache_creation_input_tokens_above_1hr"),
}
COST_COLUMN_BY_KEY = {
    CostKey.INPUT_COST: "input_cost",
    CostKey.OUTPUT_COST: "output_cost",
    CostKey.TOTAL_COST: "total_cost",
}
PROMOTED_TRACE_METADATA_KEYS = frozenset({
    TraceMetadataKey.TRACE_SESSION,
    TraceMetadataKey.TOKEN_USAGE,
    TraceMetadataKey.COST,
})

TRACE_ANALYTICS_COLUMNS_BY_METADATA_KEY = {
    TraceMetadataKey.TOKEN_USAGE: TOKEN_COLUMN_BY_KEY,
    TraceMetadataKey.COST: COST_COLUMN_BY_KEY,
}


def get_trace_analytics_metadata_filter(
    key: str, comparator: str, value: str | None, trace_info_model
) -> ColumnElement[bool]:
    columns_by_item_key = {
        item_key: getattr(trace_info_model, column)
        for item_key, column in TRACE_ANALYTICS_COLUMNS_BY_METADATA_KEY[key].items()
    }
    metadata_exists = or_(*(column.isnot(None) for column in columns_by_item_key.values()))
    if comparator == "IS NULL":
        return ~metadata_exists
    if comparator == "IS NOT NULL":
        return metadata_exists
    if comparator not in ("=", "!="):
        raise MlflowException.invalid_parameter_value(
            f"Comparator '{comparator}' is not supported for reserved metadata '{key}'. "
            "Only '=', '!=', 'IS NULL', and 'IS NOT NULL' are supported."
        )

    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        parsed = None

    converter = token_count_or_none if key == TraceMetadataKey.TOKEN_USAGE else finite_float_or_none
    normalized = (
        {
            item_key: converted
            for item_key in columns_by_item_key
            if (converted := converter(parsed.get(item_key))) is not None
        }
        if isinstance(parsed, dict) and set(parsed).issubset(columns_by_item_key)
        else {}
    )
    # Preserve equality against the exact JSON string synthesized for compatibility metadata.
    canonical_value = json.dumps(normalized)
    value_is_canonical = bool(normalized) and value == canonical_value
    if value_is_canonical:
        value_matches = and_(
            *(
                and_(column.isnot(None), column == normalized[item_key])
                if item_key in normalized
                else column.is_(None)
                for item_key, column in columns_by_item_key.items()
            )
        )
    else:
        value_matches = false()

    return value_matches if comparator == "=" else and_(metadata_exists, ~value_matches)


def validate_session_id(value: str | None) -> str | None:
    return _validate_length_limit("Session ID", MAX_CHARS_IN_TRACE_INFO_METADATA, value)


def validate_trace_name(value: str | None) -> str | None:
    return _validate_length_limit("Trace name", MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE, value)


def finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        # A magnitude too large for a float (e.g. a huge integer in cost metadata) is treated like
        # inf: non-finite, so it stores as NULL rather than crashing the batch and every rerun.
        return None
    return value if math.isfinite(value) else None


def token_count_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if (
        not value.is_finite()
        or value != value.to_integral_value()
        or not _BIGINT_MIN <= value <= _BIGINT_MAX
    ):
        return None
    return int(value)


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


def analytics_columns_from_metadata(
    metadata: dict[str, str],
) -> dict[str, float | int | str | None]:
    columns: dict[str, float | int | str | None] = {}
    if TraceMetadataKey.TRACE_SESSION in metadata:
        columns["session_id"] = validate_session_id(metadata[TraceMetadataKey.TRACE_SESSION])
    if TraceMetadataKey.TOKEN_USAGE in metadata:
        token_usage = _json_object(metadata[TraceMetadataKey.TOKEN_USAGE])
        columns.update({
            column: token_count_or_none(token_usage[key])
            for key, column in TOKEN_COLUMN_BY_KEY.items()
            if key in token_usage
        })
    if TraceMetadataKey.COST in metadata:
        cost = _json_object(metadata[TraceMetadataKey.COST])
        columns.update({
            column: finite_float_or_none(cost[key])
            for key, column in COST_COLUMN_BY_KEY.items()
            if key in cost
        })
    return columns


def compatibility_metadata_from_columns(sql_trace_info) -> dict[str, str]:
    metadata = {}
    if sql_trace_info.session_id is not None:
        metadata[TraceMetadataKey.TRACE_SESSION] = sql_trace_info.session_id

    token_usage = {
        key: token_count
        for key, column in TOKEN_COLUMN_BY_KEY.items()
        if (token_count := token_count_or_none(getattr(sql_trace_info, column))) is not None
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
        try:
            value = float(value)
        except OverflowError:
            # Too large for a float: not a usable numeric aggregate, so treat it like inf/nan.
            return None, False
        return (value, True) if math.isfinite(value) else (None, False)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "no"}:
            return (1.0 if normalized == "yes" else 0.0), False
    return None, False


def bounded_model_dimension(value: Any) -> str | None:
    return value[:MODEL_DIMENSION_MAX_LENGTH] if isinstance(value, str) else None
