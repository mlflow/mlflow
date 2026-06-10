"""Unit tests for Contains / Excludes / Matches / Equals."""

from __future__ import annotations

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.genai.scorers import Contains, Equals, Excludes, Matches


@pytest.mark.parametrize(
    ("needles", "outputs", "kwargs", "expected"),
    [
        ("workspace", "go to your workspace settings", {}, True),
        ("workspace", "go to your settings", {}, False),
        ("Workspace", "go to your WORKSPACE", {}, True),  # case-insensitive default
        ("Workspace", "go to your WORKSPACE", {"case_sensitive": True}, False),
        (["trace", "scorer"], "use trace and scorer", {}, True),
        (["trace", "scorer"], "only trace here", {}, False),
        (["trace", "scorer"], "only trace here", {"match_any": True}, True),
        (["trace", "scorer"], "no matches", {"match_any": True}, False),
    ],
)
def test_contains(needles, outputs, kwargs, expected):
    result: Feedback = Contains(needles, **kwargs)(outputs=outputs)
    assert result.value is expected
    assert result.rationale


def test_contains_field_inputs():
    result: Feedback = Contains("hello", field="inputs")(inputs="hello world", outputs="bye")
    assert result.value is True


def test_contains_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        Contains([])


def test_contains_auto_name():
    assert Contains("workspace").name == "contains_workspace"
    assert Contains(["trace", "scorer"]).name == "contains_trace_scorer"
    assert Contains(["a", "b", "c", "d", "e"]).name.endswith("_etc")


def test_contains_custom_name():
    assert Contains("workspace", name="mentions_workspace").name == "mentions_workspace"


@pytest.mark.parametrize(
    ("needles", "outputs", "kwargs", "expected"),
    [
        ("mlflow runs create", "use mlflow runs create here", {}, False),
        ("mlflow runs create", "use mlflow logs here", {}, True),
        (["foo", "bar"], "contains bar only", {}, False),
        (["foo", "bar"], "contains neither", {}, True),
    ],
)
def test_excludes(needles, outputs, kwargs, expected):
    result = Excludes(needles, **kwargs)(outputs=outputs)
    assert result.value is expected
    assert result.rationale


@pytest.mark.parametrize(
    ("pattern", "outputs", "kwargs", "expected"),
    [
        (r"docs/.+/genai", "see https://mlflow.org/docs/3.5.0/genai/intro", {}, True),
        (r"docs/.+/genai", "no docs link", {}, False),
        (r"^hello$", "hello", {}, True),
        (r"^hello$", "hello world", {}, False),
        (r"HELLO", "hello world", {}, True),  # case-insensitive default
        (r"HELLO", "hello world", {"case_sensitive": True}, False),
    ],
)
def test_matches(pattern, outputs, kwargs, expected):
    result = Matches(pattern, **kwargs)(outputs=outputs)
    assert result.value is expected
    assert result.rationale


def test_matches_invalid_pattern_raises_at_definition():
    with pytest.raises(re_error_or_value_error()):
        Matches("(unclosed")


def re_error_or_value_error():
    import re

    return re.error


@pytest.mark.parametrize(
    ("expected_val", "outputs", "kwargs", "expected_pass"),
    [
        ("yes", "yes", {}, True),
        ("yes", "YES", {}, True),  # case-insensitive default
        ("yes", "YES", {"case_sensitive": True}, False),
        ("yes", "no", {}, False),
        ("yes", "yes ", {}, False),  # exact match, no auto-trim
    ],
)
def test_equals(expected_val, outputs, kwargs, expected_pass):
    result = Equals(expected_val, **kwargs)(outputs=outputs)
    assert result.value is expected_pass
    assert result.rationale


def test_run_filters_unused_kwargs():
    # Scorer.run() filters kwargs to what __call__ accepts. Contains accepts
    # outputs + inputs but not `expectations` / `trace`, so passing those
    # extras should not raise.
    result = Contains("hi").run(outputs="say hi", expectations={"x": 1}, trace=None)
    assert result.value is True


def test_none_outputs_coerced_to_empty_string():
    # Defensive: agents that return None should not crash the scorer.
    result = Contains("anything")(outputs=None)
    assert result.value is False
