import json
from collections.abc import Callable
from decimal import Decimal
from types import SimpleNamespace

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.utils.trace_analytics import (
    analytics_columns_from_metadata,
    assessment_aggregate,
    compatibility_metadata_from_columns,
    finite_float_or_none,
    token_count_or_none,
    validate_session_id,
    validate_trace_name,
)
from mlflow.tracing.constant import (
    MAX_CHARS_IN_TRACE_INFO_METADATA,
    MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE,
    TraceMetadataKey,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (100.0, 100),
        ("100", 100),
        (Decimal("100.0"), 100),
        (-(2**63), -(2**63)),
        (2**63 - 1, 2**63 - 1),
        (None, None),
        (True, None),
        (1.5, None),
        ("1.5", None),
        (float("nan"), None),
        (float("inf"), None),
        (-(2**63) - 1, None),
        (2**63, None),
        ("not-a-number", None),
        # A huge integer stays within Decimal's arbitrary precision and is rejected by the bigint
        # range check, so it never reaches float() and never raises OverflowError.
        (10**400, None),
        (-(10**400), None),
    ],
)
def test_token_count_or_none(value, expected):
    assert token_count_or_none(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, None),
        (1.5, 1.5),
        (100, 100.0),
        ("2.5", 2.5),
        (Decimal("2.5"), 2.5),
        (float("nan"), None),
        (float("inf"), None),
        # A magnitude too large for a float must return None instead of raising OverflowError.
        (10**400, None),
        (-(10**400), None),
    ],
)
def test_finite_float_or_none(value, expected):
    assert finite_float_or_none(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, (1.0, False)),
        (False, (0.0, False)),
        (1.5, (1.5, True)),
        (3, (3.0, True)),
        ("yes", (1.0, False)),
        ("no", (0.0, False)),
        ("other", (None, False)),
        (float("nan"), (None, False)),
        (float("inf"), (None, False)),
        # A magnitude too large for a float must return (None, False) instead of raising.
        (10**400, (None, False)),
        (-(10**400), (None, False)),
    ],
)
def test_assessment_aggregate(value, expected):
    assert assessment_aggregate(value) == expected


def test_analytics_columns_from_metadata_converts_integral_token_counts():
    metadata = {
        TraceMetadataKey.TOKEN_USAGE: json.dumps({
            "input_tokens": 100,
            "output_tokens": 50.0,
            "total_tokens": "150",
            "cache_read_input_tokens": 1.5,
        })
    }

    assert analytics_columns_from_metadata(metadata) == {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "cache_read_input_tokens": None,
    }


def test_analytics_columns_from_metadata_only_converts_present_keys():
    metadata = {
        TraceMetadataKey.TOKEN_USAGE: json.dumps({"total_tokens": None}),
        TraceMetadataKey.COST: json.dumps({"input_cost": "invalid", "total_cost": 0.3}),
    }

    assert analytics_columns_from_metadata(metadata) == {
        "total_tokens": None,
        "input_cost": None,
        "total_cost": 0.3,
    }


@pytest.mark.parametrize(
    ("validator", "limit"),
    [
        (validate_session_id, MAX_CHARS_IN_TRACE_INFO_METADATA),
        (validate_trace_name, MAX_CHARS_IN_TRACE_INFO_TAGS_VALUE),
    ],
)
def test_validate_trace_dimension_length(validator: Callable[[str | None], str | None], limit: int):
    assert validator("a" * limit) == "a" * limit

    with pytest.raises(MlflowException, match=rf"maximum length of {limit}"):
        validator("a" * (limit + 1))


def test_compatibility_metadata_serializes_token_counts_as_integers():
    sql_trace_info = SimpleNamespace(
        session_id=None,
        input_tokens=100,
        output_tokens=50.0,
        total_tokens=150,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
        input_cost=None,
        output_cost=None,
        total_cost=None,
    )

    metadata = compatibility_metadata_from_columns(sql_trace_info)

    assert metadata[TraceMetadataKey.TOKEN_USAGE] == (
        '{"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}'
    )
