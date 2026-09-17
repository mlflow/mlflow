# Tests for the query-bound pagination token and max_results validation
# used by the skill registry search helpers.

from __future__ import annotations

import base64
import json

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.skill_registry_pagination import (
    SkillRegistryPaginationToken,
    validate_max_results,
)

# ---------------------------------------------------------------------------
# Token encode / decode roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filter_string", "order_by", "offset", "query_scope"),
    [
        (None, None, 0, "skills"),
        ("name LIKE '%review%'", ["name ASC"], 100, "skills"),
        ("status = 'active'", ["creation_timestamp DESC"], 200, "skill_versions:acme/code-review"),
        (None, None, 0, "agent_plugins"),
        ("member_name = 'review'", None, 50, "agent_plugin_versions:acme/my-plugin"),
    ],
)
def test_encode_decode_roundtrip(filter_string, order_by, offset, query_scope):
    token = SkillRegistryPaginationToken(
        filter_string=filter_string,
        order_by=order_by,
        offset=offset,
        query_scope=query_scope,
    )
    encoded = token.encode()
    decoded = SkillRegistryPaginationToken.decode(encoded)
    assert decoded == token


# ---------------------------------------------------------------------------
# Query binding validation
# ---------------------------------------------------------------------------


def test_validate_accepts_matching_query():
    token = SkillRegistryPaginationToken(
        filter_string="status = 'active'",
        order_by=["name ASC"],
        offset=0,
        query_scope="skills",
    )
    token.validate(
        filter_string="status = 'active'",
        order_by=["name ASC"],
        query_scope="skills",
    )


def test_validate_rejects_different_filter_string():
    token = SkillRegistryPaginationToken(
        filter_string="status = 'active'",
        order_by=None,
        offset=0,
        query_scope="skills",
    )
    with pytest.raises(MlflowException, match=r"(?i)different filter_string"):
        token.validate(
            filter_string="status = 'draft'",
            order_by=None,
            query_scope="skills",
        )


def test_validate_rejects_different_order_by():
    token = SkillRegistryPaginationToken(
        filter_string=None,
        order_by=["name ASC"],
        offset=0,
        query_scope="skills",
    )
    with pytest.raises(MlflowException, match=r"(?i)different order_by"):
        token.validate(
            filter_string=None,
            order_by=["name DESC"],
            query_scope="skills",
        )


def test_validate_rejects_different_query_scope():
    token = SkillRegistryPaginationToken(
        filter_string=None,
        order_by=None,
        offset=0,
        query_scope="skills",
    )
    with pytest.raises(MlflowException, match=r"(?i)different query scope"):
        token.validate(
            filter_string=None,
            order_by=None,
            query_scope="agent_plugins",
        )


# ---------------------------------------------------------------------------
# Malformed tokens
# ---------------------------------------------------------------------------


def test_decode_rejects_invalid_base64():
    with pytest.raises(MlflowException, match=r"(?i)could not decode"):
        SkillRegistryPaginationToken.decode("not-valid-base64!!!")


def test_decode_rejects_invalid_json():
    bad = base64.b64encode(b"not json").decode()
    with pytest.raises(MlflowException, match=r"(?i)could not decode"):
        SkillRegistryPaginationToken.decode(bad)


def test_decode_rejects_missing_fields():
    incomplete = base64.b64encode(json.dumps({"offset": 0}).encode()).decode()
    with pytest.raises(MlflowException, match=r"(?i)missing or malformed"):
        SkillRegistryPaginationToken.decode(incomplete)


def test_decode_rejects_negative_offset():
    payload = {
        "filter_string": None,
        "order_by": None,
        "offset": -1,
        "query_scope": "skills",
    }
    token = base64.b64encode(json.dumps(payload).encode()).decode()
    with pytest.raises(MlflowException, match=r"(?i)non-negative integer"):
        SkillRegistryPaginationToken.decode(token)


def test_decode_rejects_non_integer_offset():
    payload = {
        "filter_string": None,
        "order_by": None,
        "offset": "abc",
        "query_scope": "skills",
    }
    token = base64.b64encode(json.dumps(payload).encode()).decode()
    with pytest.raises(MlflowException, match=r"(?i)non-negative integer"):
        SkillRegistryPaginationToken.decode(token)


# ---------------------------------------------------------------------------
# validate_max_results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [1, 50, 100, 1000])
def test_validate_max_results_accepts_valid(value):
    validate_max_results(value)


@pytest.mark.parametrize("value", [0, -1, -100])
def test_validate_max_results_rejects_non_positive(value):
    with pytest.raises(MlflowException, match=r"(?i)at least 1"):
        validate_max_results(value)


def test_validate_max_results_rejects_over_threshold():
    with pytest.raises(MlflowException, match=r"(?i)at most 1000"):
        validate_max_results(1001)


def test_validate_max_results_with_custom_threshold():
    validate_max_results(50, threshold=50)
    with pytest.raises(MlflowException, match=r"(?i)at most 50"):
        validate_max_results(51, threshold=50)
