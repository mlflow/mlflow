"""Frozen pure conversion helpers for Alembic revision 75868b020152.

IMMUTABLE: this module defines the historical trace-analytics backfill contract shared by the
offline migration and its prepopulation command. Do not edit after the revision ships; later
analytics changes need a new revision-specific companion module.
"""

import json
import math
from decimal import Decimal, InvalidOperation
from typing import Any

_BIGINT_MIN = -(2**63)
_BIGINT_MAX = 2**63 - 1
SESSION_ID_MAX_LENGTH = 250

TOKEN_COLUMNS = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "total_tokens": "total_tokens",
    "cache_read_input_tokens": "cache_read_input_tokens",
    "cache_creation_input_tokens": "cache_creation_input_tokens",
    "cache_creation_input_tokens_above_1hr": "cache_creation_input_tokens_above_1hr",
}
COST_COLUMNS = {
    "input_cost": "input_cost",
    "output_cost": "output_cost",
    "total_cost": "total_cost",
}

_TRACE_SESSION_METADATA_KEY = "mlflow.trace.session"
_TOKEN_USAGE_METADATA_KEY = "mlflow.trace.tokenUsage"
_COST_METADATA_KEY = "mlflow.trace.cost"


def json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
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


def trace_analytics_values(
    metadata: dict[str, Any], token_metrics: dict[str, Any]
) -> dict[str, float | int | str | None]:
    """Return this revision's promoted trace values with metric values taking precedence."""
    token_usage = json_object(metadata.get(_TOKEN_USAGE_METADATA_KEY))
    cost = json_object(metadata.get(_COST_METADATA_KEY))
    return {
        "session_id": bounded_string_or_none(
            metadata.get(_TRACE_SESSION_METADATA_KEY), SESSION_ID_MAX_LENGTH
        ),
        **{
            column: token_count_or_none(token_metrics.get(column, token_usage.get(key)))
            for key, column in TOKEN_COLUMNS.items()
        },
        **{column: finite_float_or_none(cost.get(key)) for key, column in COST_COLUMNS.items()},
    }


def bounded_string_or_none(value: Any, max_length: int) -> str | None:
    return value[:max_length] if isinstance(value, str) else None
