# Tests for the search-text projection builder used by the skill registry.

from __future__ import annotations

from mlflow.store.tracking.skill_registry_search_text import (
    build_skill_search_text,
)

# ---------------------------------------------------------------------------
# build_skill_search_text
# ---------------------------------------------------------------------------


def test_skill_search_text_name_only():
    assert build_skill_search_text("code-review") == "code-review"


def test_skill_search_text_name_and_description():
    result = build_skill_search_text("code-review", description="Reviews pull requests")
    assert result == "code-review Reviews pull requests"


def test_skill_search_text_none_description():
    result = build_skill_search_text("code-review", description=None)
    assert result == "code-review"


def test_skill_search_text_empty_description():
    result = build_skill_search_text("code-review", description="")
    assert result == "code-review"


def test_skill_search_text_normalizes_whitespace():
    result = build_skill_search_text("name", description="  extra   spaces  ")
    assert result == "name extra spaces"
