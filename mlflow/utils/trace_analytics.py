import math
from decimal import Decimal, InvalidOperation
from typing import Any

_BIGINT_MIN = -(2**63)
_BIGINT_MAX = 2**63 - 1


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
