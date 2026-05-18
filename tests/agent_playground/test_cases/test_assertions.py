import pytest

from mlflow.agent_playground.test_cases.assertions import (
    AssertionResult,
    evaluate_assertions,
)
from mlflow.agent_playground.test_cases.entities import AssertionExpectations


def test_empty_spec_passes_for_any_input():
    result = evaluate_assertions(
        AssertionExpectations(),
        final_response_text="anything goes",
        tool_call_names=["any_tool"],
    )
    assert result == AssertionResult(outcome="pass", reasons=())


def test_must_contain_pass():
    result = evaluate_assertions(
        AssertionExpectations(must_contain=["log level"]),
        final_response_text="Set the log level to INFO",
        tool_call_names=[],
    )
    assert result.outcome == "pass"
    assert result.reasons == ()


def test_must_contain_fail():
    result = evaluate_assertions(
        AssertionExpectations(must_contain=["log level", "DEBUG"]),
        final_response_text="Set the log level to INFO",
        tool_call_names=[],
    )
    assert result.outcome == "fail"
    assert result.reasons == ("must_contain: 'DEBUG' was not present in the response",)


def test_must_not_contain_pass():
    result = evaluate_assertions(
        AssertionExpectations(must_not_contain=["error"]),
        final_response_text="All good",
        tool_call_names=[],
    )
    assert result.outcome == "pass"


def test_must_not_contain_fail():
    result = evaluate_assertions(
        AssertionExpectations(must_not_contain=["error"]),
        final_response_text="There was an error",
        tool_call_names=[],
    )
    assert result.outcome == "fail"
    assert result.reasons == ("must_not_contain: 'error' was present in the response",)


def test_must_call_tool_pass():
    result = evaluate_assertions(
        AssertionExpectations(must_call_tool=["search_docs"]),
        final_response_text="",
        tool_call_names=["search_docs", "other_tool"],
    )
    assert result.outcome == "pass"


def test_must_call_tool_fail():
    result = evaluate_assertions(
        AssertionExpectations(must_call_tool=["search_docs"]),
        final_response_text="",
        tool_call_names=["other_tool"],
    )
    assert result.outcome == "fail"
    assert result.reasons == ("must_call_tool: 'search_docs' was not invoked",)


def test_must_not_call_tool_pass():
    result = evaluate_assertions(
        AssertionExpectations(must_not_call_tool=["delete_record"]),
        final_response_text="",
        tool_call_names=["search_docs"],
    )
    assert result.outcome == "pass"


def test_must_not_call_tool_fail():
    result = evaluate_assertions(
        AssertionExpectations(must_not_call_tool=["delete_record"]),
        final_response_text="",
        tool_call_names=["delete_record"],
    )
    assert result.outcome == "fail"
    assert result.reasons == ("must_not_call_tool: 'delete_record' was invoked",)


def test_multiple_failures_listed_separately():
    result = evaluate_assertions(
        AssertionExpectations(
            must_contain=["log level"],
            must_call_tool=["search_docs"],
        ),
        final_response_text="some other text",
        tool_call_names=[],
    )
    assert result.outcome == "fail"
    assert len(result.reasons) == 2


def test_all_four_clauses_pass_together():
    result = evaluate_assertions(
        AssertionExpectations(
            must_contain=["log level"],
            must_not_contain=["error"],
            must_call_tool=["search_docs"],
            must_not_call_tool=["delete_record"],
        ),
        final_response_text="Set the log level to INFO. All good.",
        tool_call_names=["search_docs", "format_response"],
    )
    assert result.outcome == "pass"
    assert result.reasons == ()


@pytest.mark.parametrize(
    ("needle", "haystack", "expected_outcome"),
    [
        ("foo", "foo", "pass"),
        ("foo", "FOO", "fail"),
        ("foo bar", "this foo bar is here", "pass"),
        ("foo", "fobar", "fail"),
    ],
)
def test_must_contain_substring_semantics(needle: str, haystack: str, expected_outcome: str):
    result = evaluate_assertions(
        AssertionExpectations(must_contain=[needle]),
        final_response_text=haystack,
        tool_call_names=[],
    )
    assert result.outcome == expected_outcome


def test_tool_call_set_semantics_at_least_once():
    # Multiple invocations of the same tool count as one set member;
    # ``must_call_tool=["search_docs"]`` holds whether the tool was
    # called once or N times.
    result = evaluate_assertions(
        AssertionExpectations(must_call_tool=["search_docs"]),
        final_response_text="",
        tool_call_names=["search_docs", "search_docs", "search_docs"],
    )
    assert result.outcome == "pass"


def test_assertion_result_pass_with_reasons_rejected():
    with pytest.raises(ValueError, match="must not carry reasons"):
        AssertionResult(outcome="pass", reasons=("nope",))


def test_assertion_result_fail_without_reasons_rejected():
    with pytest.raises(ValueError, match="must carry at least one reason"):
        AssertionResult(outcome="fail")


@pytest.mark.parametrize(
    ("field_name", "blank_value"),
    [
        ("must_contain", ""),
        ("must_contain", "   "),
        ("must_not_contain", ""),
        ("must_call_tool", "\t"),
        ("must_not_call_tool", ""),
    ],
    ids=[
        "must_contain-empty",
        "must_contain-whitespace",
        "must_not_contain-empty",
        "must_call_tool-tab",
        "must_not_call_tool-empty",
    ],
)
def test_assertion_expectations_rejects_blank_clauses(field_name: str, blank_value: str):
    with pytest.raises(ValueError, match="must be non-empty"):
        AssertionExpectations(**{field_name: [blank_value]})
