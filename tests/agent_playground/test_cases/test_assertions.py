import pytest

from mlflow.agent_playground.test_cases.assertions import (
    AssertionResult,
    AssertionSpec,
    evaluate_assertions,
)


def test_empty_spec_passes_for_any_input():
    result = evaluate_assertions(
        AssertionSpec(),
        final_response_text="anything goes",
        tool_call_names=["any_tool"],
    )
    assert result == AssertionResult(passed=True, reasons=())


def test_must_contain_pass():
    result = evaluate_assertions(
        AssertionSpec(must_contain=["log level"]),
        final_response_text="Set the log level to INFO",
        tool_call_names=[],
    )
    assert result.passed
    assert result.reasons == ()


def test_must_contain_fail():
    result = evaluate_assertions(
        AssertionSpec(must_contain=["log level", "DEBUG"]),
        final_response_text="Set the log level to INFO",
        tool_call_names=[],
    )
    assert not result.passed
    assert any("DEBUG" in reason for reason in result.reasons)


def test_must_not_contain_pass():
    result = evaluate_assertions(
        AssertionSpec(must_not_contain=["error"]),
        final_response_text="All good",
        tool_call_names=[],
    )
    assert result.passed


def test_must_not_contain_fail():
    result = evaluate_assertions(
        AssertionSpec(must_not_contain=["error"]),
        final_response_text="There was an error",
        tool_call_names=[],
    )
    assert not result.passed
    assert any("error" in reason for reason in result.reasons)


def test_must_call_tool_pass():
    result = evaluate_assertions(
        AssertionSpec(must_call_tool=["search_docs"]),
        final_response_text="",
        tool_call_names=["search_docs", "other_tool"],
    )
    assert result.passed


def test_must_call_tool_fail():
    result = evaluate_assertions(
        AssertionSpec(must_call_tool=["search_docs"]),
        final_response_text="",
        tool_call_names=["other_tool"],
    )
    assert not result.passed
    assert any("search_docs" in reason for reason in result.reasons)


def test_must_not_call_tool_pass():
    result = evaluate_assertions(
        AssertionSpec(must_not_call_tool=["delete_record"]),
        final_response_text="",
        tool_call_names=["search_docs"],
    )
    assert result.passed


def test_must_not_call_tool_fail():
    result = evaluate_assertions(
        AssertionSpec(must_not_call_tool=["delete_record"]),
        final_response_text="",
        tool_call_names=["delete_record"],
    )
    assert not result.passed
    assert any("delete_record" in reason for reason in result.reasons)


def test_multiple_failures_listed_separately():
    result = evaluate_assertions(
        AssertionSpec(
            must_contain=["log level"],
            must_call_tool=["search_docs"],
        ),
        final_response_text="some other text",
        tool_call_names=[],
    )
    assert not result.passed
    assert len(result.reasons) == 2


def test_all_four_clauses_pass_together():
    result = evaluate_assertions(
        AssertionSpec(
            must_contain=["log level"],
            must_not_contain=["error"],
            must_call_tool=["search_docs"],
            must_not_call_tool=["delete_record"],
        ),
        final_response_text="Set the log level to INFO. All good.",
        tool_call_names=["search_docs", "format_response"],
    )
    assert result.passed
    assert result.reasons == ()


@pytest.mark.parametrize(
    ("needle", "haystack", "expected_pass"),
    [
        ("foo", "foo", True),
        ("foo", "FOO", False),
        ("foo bar", "this foo bar is here", True),
        ("foo", "fobar", False),
    ],
)
def test_must_contain_substring_semantics(needle: str, haystack: str, expected_pass: bool):
    result = evaluate_assertions(
        AssertionSpec(must_contain=[needle]),
        final_response_text=haystack,
        tool_call_names=[],
    )
    assert result.passed == expected_pass
