# Tests for the search-text projection builders and ORM recompute wrappers
# used by the skill registry.

from __future__ import annotations

from unittest.mock import Mock

import pytest

from mlflow.store.tracking.skill_registry_search_text import (
    build_agent_plugin_version_search_text,
    build_skill_search_text,
    recompute_agent_plugin_version_search_text,
    recompute_skill_search_text,
)

# ---------------------------------------------------------------------------
# build_skill_search_text
# ---------------------------------------------------------------------------


def test_skill_search_text_name_only():
    assert build_skill_search_text("code-review") == "code-review"


def test_skill_search_text_name_and_description():
    result = build_skill_search_text("code-review", description="Reviews pull requests")
    assert result == "code-review Reviews pull requests"


def test_skill_search_text_with_keywords():
    result = build_skill_search_text(
        "code-review",
        description="Reviews PRs",
        keywords=["security", "lint"],
    )
    assert result == "code-review Reviews PRs security lint"


def test_skill_search_text_none_description():
    result = build_skill_search_text("code-review", description=None)
    assert result == "code-review"


def test_skill_search_text_empty_description():
    result = build_skill_search_text("code-review", description="")
    assert result == "code-review"


def test_skill_search_text_empty_keywords():
    result = build_skill_search_text("code-review", description="desc", keywords=[])
    assert result == "code-review desc"


def test_skill_search_text_normalizes_whitespace():
    result = build_skill_search_text("name", description="  extra   spaces  ")
    assert result == "name extra spaces"


# ---------------------------------------------------------------------------
# build_agent_plugin_version_search_text
# ---------------------------------------------------------------------------


def test_agent_plugin_search_text_all_fields():
    result = build_agent_plugin_version_search_text(
        name="my-plugin",
        parent_description="A helpful plugin",
        organization="acme",
        manifest_description="Does things",
        keywords=["automation", "deploy"],
        author_name="Jane Smith",
    )
    assert result == "my-plugin A helpful plugin acme Does things automation deploy Jane Smith"


def test_agent_plugin_search_text_minimal():
    result = build_agent_plugin_version_search_text(name="my-plugin")
    assert result == "my-plugin"


def test_agent_plugin_search_text_none_fields():
    result = build_agent_plugin_version_search_text(
        name="my-plugin",
        parent_description=None,
        organization="",
        manifest_description=None,
        keywords=None,
        author_name=None,
    )
    assert result == "my-plugin"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"name": "p", "parent_description": "desc", "organization": "org"},
            "p desc org",
        ),
        (
            {"name": "p", "keywords": ["k1", "k2"]},
            "p k1 k2",
        ),
        (
            {"name": "p", "author_name": "Bob"},
            "p Bob",
        ),
    ],
)
def test_agent_plugin_search_text_partial_fields(kwargs, expected):
    assert build_agent_plugin_version_search_text(**kwargs) == expected


def test_agent_plugin_search_text_normalizes_whitespace():
    result = build_agent_plugin_version_search_text(
        name="  my-plugin  ",
        manifest_description="  lots   of    spaces  ",
    )
    assert result == "my-plugin lots of spaces"


# ---------------------------------------------------------------------------
# recompute_* ORM convenience wrappers
# ---------------------------------------------------------------------------


def test_recompute_skill_search_text():
    row = Mock()
    row.name = "code-review"
    row.description = "Reviews PRs"
    assert recompute_skill_search_text(row) == "code-review Reviews PRs"
    assert recompute_skill_search_text(row, keywords=["security"]) == (
        "code-review Reviews PRs security"
    )


def test_recompute_agent_plugin_version_search_text():
    row = Mock()
    row.name = "my-plugin"
    row.organization = "acme"
    row.plugin_json = {
        "description": "Plugin description",
        "keywords": ["deploy", "ci"],
        "author": {"name": "Jane"},
    }
    result = recompute_agent_plugin_version_search_text(
        row,
        parent_description="Parent desc",
    )
    assert result == "my-plugin Parent desc acme Plugin description deploy ci Jane"


def test_recompute_agent_plugin_version_search_text_no_author():
    row = Mock()
    row.name = "my-plugin"
    row.organization = ""
    row.plugin_json = {"description": "desc"}
    result = recompute_agent_plugin_version_search_text(row)
    assert result == "my-plugin desc"


def test_recompute_agent_plugin_version_search_text_none_plugin_json():
    row = Mock()
    row.name = "my-plugin"
    row.organization = ""
    row.plugin_json = None
    result = recompute_agent_plugin_version_search_text(row)
    assert result == "my-plugin"
