from __future__ import annotations

import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.search_utils import (
    SearchAgentPluginUtils,
    SearchAgentPluginVersionUtils,
    SearchSkillUtils,
    SearchSkillVersionUtils,
    SearchUtils,
)
from mlflow.utils.validation import _validate_skill_tag

# ---------------------------------------------------------------------------
# SearchSkillUtils — valid filters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filter_string", "expected_key", "expected_comparator"),
    [
        ("name = 'code-review'", "name", "="),
        ("name LIKE '%review%'", "name", "LIKE"),
        ("name ILIKE '%Review%'", "name", "ILIKE"),
        ("status = 'active'", "status", "="),
        ("organization = 'acme'", "organization", "="),
        ("description LIKE '%security%'", "description", "LIKE"),
        ("search_text LIKE '%review%'", "search_text", "LIKE"),
    ],
)
def test_skill_filter_parses_valid_expression(filter_string, expected_key, expected_comparator):
    parsed = SearchSkillUtils.parse_search_filter(filter_string)
    assert len(parsed) == 1
    assert parsed[0]["type"] == "attribute"
    assert parsed[0]["key"] == expected_key
    assert parsed[0]["comparator"].upper() == expected_comparator


def test_skill_filter_parses_tag():
    parsed = SearchSkillUtils.parse_search_filter("tags.team = 'platform'")
    assert len(parsed) == 1
    assert parsed[0]["type"] == "tag"
    assert parsed[0]["key"] == "team"
    assert parsed[0]["value"] == "platform"


@pytest.mark.parametrize(
    ("filter_string", "expected"),
    [
        (
            "status = 'active' AND organization = 'acme'",
            [("attribute", "status", "=", "active"), ("attribute", "organization", "=", "acme")],
        ),
        (
            "organization = 'acme' AND tags.team LIKE 'plat%'",
            [("attribute", "organization", "=", "acme"), ("tag", "team", "LIKE", "plat%")],
        ),
    ],
)
def test_skill_filter_parses_compound(filter_string, expected):
    parsed = SearchSkillUtils.parse_search_filter(filter_string)
    assert [(f["type"], f["key"], f["comparator"], f["value"]) for f in parsed] == expected


def test_skill_filter_parses_organization_like():
    parsed = SearchSkillUtils.parse_search_filter("organization LIKE '%acme%'")
    assert len(parsed) == 1
    assert parsed[0]["key"] == "organization"
    assert parsed[0]["comparator"].upper() == "LIKE"


def test_skill_filter_parses_empty_filter():
    assert SearchSkillUtils.parse_search_filter(None) == []
    assert SearchSkillUtils.parse_search_filter("") == []


def test_skill_filter_preserves_double_quoted_organization_in_value():
    parsed = SearchSkillUtils.parse_search_filter('description LIKE "%organization in GitHub%"')
    assert parsed[0]["value"] == "%organization in GitHub%"


def test_skill_filter_preserves_dotted_tag_key():
    parsed = SearchSkillUtils.parse_search_filter("tags.mlflow.organization = 'acme'")
    assert parsed[0]["key"] == "mlflow.organization"


_ALL_SEARCH_UTILS = [
    SearchSkillUtils,
    SearchSkillVersionUtils,
    SearchAgentPluginUtils,
    SearchAgentPluginVersionUtils,
]


@pytest.mark.parametrize("utils", _ALL_SEARCH_UTILS)
@pytest.mark.parametrize("key", ["my organization in GitHub", "API version in use"])
def test_filter_preserves_backtick_quoted_tag_key_containing_keyword(utils, key):
    _validate_skill_tag(key, "yes")
    filter_string = f"tags.`{key}` = 'yes'"
    assert SearchUtils.parse_search_filter(filter_string)[0]["key"] == key
    assert utils.parse_search_filter(filter_string) == [
        {"type": "tag", "key": key, "comparator": "=", "value": "yes"}
    ]


@pytest.mark.parametrize("utils", _ALL_SEARCH_UTILS)
def test_filter_preserves_value_with_escaped_newline(utils):
    # A backslash before a newline must not end the quoted value early, and the
    # keyword field after it must still be quoted.
    value = "%a\\\nb organization = x%"
    parsed = utils.parse_search_filter(f"tags.note LIKE '{value}' AND organization = 'acme'")
    assert parsed == [
        {"type": "tag", "key": "note", "comparator": "LIKE", "value": value},
        {"type": "attribute", "key": "organization", "comparator": "=", "value": "acme"},
    ]


def test_skill_filter_rejects_unsupported_field():
    with pytest.raises(MlflowException, match=r"(?i)invalid"):
        SearchSkillUtils.parse_search_filter("nonexistent = 'val'")


@pytest.mark.parametrize(
    ("filter_string", "match"),
    [
        # Range operators are not part of the tag or string-attribute vocabulary.
        ("tags.team > 'a'", r"Invalid comparator '>' for tag 'team'"),
        ("tags.team <= 'a'", r"Invalid comparator '<=' for tag 'team'"),
        ("name > 'a'", r"Invalid comparator '>' for attribute 'name'"),
        ("status >= 'active'", r"Invalid comparator '>=' for attribute 'status'"),
        # IS NULL is excluded from the registry tag vocabulary.
        ("tags.team IS NULL", r"Invalid comparator 'IS NULL' for tag 'team'"),
        # Run-search entity types are not part of the registry grammar.
        ("params.foo = 'x'", r"Invalid filter type 'parameter'"),
        ("metrics.m > 1", r"Invalid filter type 'metric'"),
    ],
)
def test_skill_filter_rejects_unsupported_operator(filter_string, match):
    with pytest.raises(MlflowException, match=match) as exc:
        SearchSkillUtils.parse_search_filter(filter_string)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


@pytest.mark.parametrize(
    ("filter_string", "expected"),
    [
        ("name like '%review%'", "LIKE"),
        ("name Ilike '%review%'", "ILIKE"),
        ("tags.team like 'plat%'", "LIKE"),
    ],
)
def test_skill_filter_normalizes_comparator_case(filter_string, expected):
    assert SearchSkillUtils.parse_search_filter(filter_string)[0]["comparator"] == expected


@pytest.mark.parametrize(
    ("filter_string", "comparator", "values"),
    [
        ("status IN ('active', 'draft')", "IN", ("active", "draft")),
        ("status NOT IN ('deleted')", "NOT IN", ("deleted",)),
    ],
)
def test_skill_filter_parses_status_list(filter_string, comparator, values):
    parsed = SearchSkillVersionUtils.parse_search_filter(filter_string)
    assert parsed[0]["comparator"] == comparator
    assert parsed[0]["value"] == values


def test_skill_filter_rejects_list_on_non_status_field():
    with pytest.raises(MlflowException, match=r"Only 'status' supports IN") as exc:
        SearchSkillUtils.parse_search_filter("organization IN ('acme', 'beta')")
    assert "run_id" not in exc.value.message


# ---------------------------------------------------------------------------
# SearchSkillVersionUtils — valid filters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filter_string", "expected_key"),
    [
        ("status = 'active'", "status"),
        ("organization = 'acme'", "organization"),
        ("source_type = 'git'", "source_type"),
        ("digest = 'abc123def456'", "digest"),
    ],
)
def test_skill_version_filter_parses_valid_expression(filter_string, expected_key):
    parsed = SearchSkillVersionUtils.parse_search_filter(filter_string)
    assert len(parsed) == 1
    assert parsed[0]["key"] == expected_key


def test_skill_version_filter_rejects_name():
    with pytest.raises(MlflowException, match=r"(?i)invalid"):
        SearchSkillVersionUtils.parse_search_filter("name = 'code-review'")


# ---------------------------------------------------------------------------
# SearchAgentPluginUtils — valid filters
# ---------------------------------------------------------------------------


def test_agent_plugin_filter_parses_member_name():
    parsed = SearchAgentPluginUtils.parse_search_filter("member_name = 'code-review'")
    assert len(parsed) == 1
    assert parsed[0]["key"] == "member_name"
    assert parsed[0]["value"] == "code-review"


@pytest.mark.parametrize(
    "filter_string",
    [
        "member_name LIKE 'code%'",
        "member_name ILIKE 'CODE%'",
        "member_name != 'code-review'",
    ],
)
def test_agent_plugin_filter_rejects_non_equality_member_name(filter_string):
    # Member search matches membership rows by exact name, so only = is supported.
    with pytest.raises(MlflowException, match=r"Supported comparators: \['='\]") as exc:
        SearchAgentPluginUtils.parse_search_filter(filter_string)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_agent_plugin_filter_parses_search_text():
    parsed = SearchAgentPluginUtils.parse_search_filter("search_text ILIKE '%deploy%'")
    assert len(parsed) == 1
    assert parsed[0]["key"] == "search_text"


# ---------------------------------------------------------------------------
# SearchAgentPluginVersionUtils — valid filters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filter_string", "expected_key"),
    [
        ("status = 'active'", "status"),
        ("organization = 'acme'", "organization"),
        ("source_type = 'oci'", "source_type"),
    ],
)
def test_agent_plugin_version_filter_parses_valid(filter_string, expected_key):
    parsed = SearchAgentPluginVersionUtils.parse_search_filter(filter_string)
    assert len(parsed) == 1
    assert parsed[0]["key"] == expected_key


def test_agent_plugin_version_filter_rejects_member_name():
    with pytest.raises(MlflowException, match=r"(?i)invalid"):
        SearchAgentPluginVersionUtils.parse_search_filter("member_name = 'review'")


# ---------------------------------------------------------------------------
# Value shape: only IN and NOT IN take a list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utils", _ALL_SEARCH_UTILS)
@pytest.mark.parametrize(
    "filter_string",
    [
        "status = ('active')",
        "status != ('active', 'draft')",
        "status LIKE ('act%')",
        "status ILIKE ('ACT%')",
    ],
)
def test_filter_rejects_parenthesized_value_for_scalar_comparator(utils, filter_string):
    # A parenthesized value parses as a tuple; comparing a column to one fails in
    # the database, so it is rejected here instead.
    with pytest.raises(MlflowException, match=r"requires a single value") as exc:
        utils.parse_search_filter(filter_string)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


@pytest.mark.parametrize("utils", _ALL_SEARCH_UTILS)
@pytest.mark.parametrize("comparator", ["IN", "NOT IN"])
def test_filter_accepts_list_for_list_comparators(utils, comparator):
    parsed = utils.parse_search_filter(f"status {comparator} ('active', 'draft')")
    assert parsed == [
        {
            "type": "attribute",
            "key": "status",
            "comparator": comparator,
            "value": ("active", "draft"),
        }
    ]


@pytest.mark.parametrize("utils", _ALL_SEARCH_UTILS)
@pytest.mark.parametrize("filter_string", ["status IN 'active'", "status NOT IN 'active'"])
def test_filter_rejects_scalar_value_for_list_comparators(utils, filter_string):
    # The base parser rejects this shape before the registry check runs, so the
    # message comes from there rather than from the value-shape rule.
    with pytest.raises(MlflowException, match=r"(?i)invalid clause") as exc:
        utils.parse_search_filter(filter_string)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


@pytest.mark.parametrize(
    ("filter_string", "match"),
    [
        # Field-specific rules still report their own message rather than the
        # generic value-shape one.
        ("name IN ('a', 'b')", r"Only 'status' supports IN and NOT IN"),
        ("tags.team IN ('a')", r"(?i)quoted string value for tag"),
    ],
)
def test_filter_keeps_field_specific_list_errors(filter_string, match):
    with pytest.raises(MlflowException, match=match):
        SearchSkillUtils.parse_search_filter(filter_string)


# ---------------------------------------------------------------------------
# Numeric attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("utils", _ALL_SEARCH_UTILS)
@pytest.mark.parametrize("key", ["created_at", "last_updated_at"])
@pytest.mark.parametrize("comparator", ["=", "!=", ">", ">=", "<", "<="])
def test_filter_parses_timestamp_comparisons(utils, key, comparator):
    parsed = utils.parse_search_filter(f"{key} {comparator} 1700000000000")
    assert parsed == [
        {
            "type": "attribute",
            "key": key,
            "comparator": comparator,
            "value": 1700000000000,
        }
    ]


@pytest.mark.parametrize("utils", _ALL_SEARCH_UTILS)
@pytest.mark.parametrize("key", ["created_at", "last_updated_at"])
def test_filter_rejects_non_numeric_timestamp_value(utils, key):
    with pytest.raises(MlflowException, match=r"(?i)numeric value"):
        utils.parse_search_filter(f"{key} LIKE '17%'")
