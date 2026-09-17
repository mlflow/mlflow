from __future__ import annotations

import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.search_utils import (
    SearchAgentPluginUtils,
    SearchAgentPluginVersionUtils,
    SearchSkillUtils,
    SearchSkillVersionUtils,
)

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


def test_skill_filter_parses_compound():
    parsed = SearchSkillUtils.parse_search_filter("status = 'active' AND organization = 'acme'")
    assert len(parsed) == 2


def test_skill_filter_parses_organization_like():
    parsed = SearchSkillUtils.parse_search_filter("organization LIKE '%acme%'")
    assert len(parsed) == 1
    assert parsed[0]["key"] == "organization"
    assert parsed[0]["comparator"].upper() == "LIKE"


def test_skill_filter_parses_empty_filter():
    assert SearchSkillUtils.parse_search_filter(None) == []
    assert SearchSkillUtils.parse_search_filter("") == []


def test_skill_filter_rejects_unsupported_field():
    with pytest.raises(MlflowException, match=r"(?i)invalid"):
        SearchSkillUtils.parse_search_filter("nonexistent = 'val'")


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
