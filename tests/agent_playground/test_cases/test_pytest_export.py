import ast

import pytest

from mlflow.agent_playground.test_cases.entities import (
    AssertionSpec,
    JudgeSpec,
    PersonaSpec,
    TestCaseRow,
    TestSpec,
)
from mlflow.agent_playground.test_cases.pytest_export import render_pytest_suite


def _assertion_case(test_case_id: str, **overrides) -> TestCaseRow:
    spec = TestSpec(
        strategy="assertion",
        rationale_summary="cite docs",
        assertion=AssertionSpec(must_contain=["docs"]),
    )
    payload = {
        "test_case_id": test_case_id,
        "spec": spec,
        "conversation_messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(overrides)
    return TestCaseRow(**payload)


def test_renders_valid_python_for_empty_suite():
    source = render_pytest_suite("exp-1", [])
    ast.parse(source)


def test_renders_valid_python_for_assertion_only_suite():
    cases = [_assertion_case(f"tc-{i:03d}") for i in range(3)]
    source = render_pytest_suite("exp-1", cases)
    ast.parse(source)


def test_header_records_included_and_excluded_counts():
    cases = [
        _assertion_case("tc-001"),
        _assertion_case(
            "tc-002",
            spec=TestSpec(
                strategy="judge",
                rationale_summary="r",
                judge=JudgeSpec(criteria="be nice"),
            ),
        ),
        _assertion_case(
            "tc-003",
            spec=TestSpec(
                strategy="assertion",
                rationale_summary="r",
                assertion=AssertionSpec(must_contain=["x"]),
                persona=PersonaSpec(goal="g"),
            ),
        ),
        _assertion_case("tc-004", conversation_messages=[]),
    ]
    source = render_pytest_suite("exp-1", cases)
    assert "Cases included: 1" in source
    assert "Cases excluded: 3" in source
    assert "judge: 1" in source
    assert "persona: 1" in source
    assert "no-conversation: 1" in source


def test_assertion_payload_is_present():
    case = _assertion_case("tc-abc")
    source = render_pytest_suite("exp-1", [case])
    assert '"test_case_id": "tc-abc"' in source
    assert '"must_contain": [' in source
    assert '"docs"' in source


def test_judge_case_excluded_from_payload():
    judge_case = TestCaseRow(
        test_case_id="tc-judge",
        spec=TestSpec(
            strategy="judge",
            rationale_summary="r",
            judge=JudgeSpec(criteria="be helpful"),
        ),
        conversation_messages=[{"role": "user", "content": "hi"}],
    )
    source = render_pytest_suite("exp-1", [judge_case])
    assert "tc-judge" not in source


def test_persona_case_excluded_from_payload():
    case = _assertion_case(
        "tc-persona",
        spec=TestSpec(
            strategy="assertion",
            rationale_summary="r",
            assertion=AssertionSpec(must_contain=["x"]),
            persona=PersonaSpec(goal="learn", persona="curious dev"),
        ),
    )
    source = render_pytest_suite("exp-1", [case])
    assert "tc-persona" not in source


def test_no_messages_case_excluded_from_payload():
    case = _assertion_case("tc-empty", conversation_messages=[])
    source = render_pytest_suite("exp-1", [case])
    assert "tc-empty" not in source


def test_includes_agent_url_default():
    source = render_pytest_suite("exp-1", [])
    assert "http://localhost:8000/invocations" in source
    assert "MLFLOW_AGENT_URL" in source


def test_includes_timeout_env_var():
    source = render_pytest_suite("exp-1", [])
    assert "MLFLOW_AGENT_TIMEOUT_SECONDS" in source


@pytest.mark.parametrize(
    ("must_field", "value"),
    [
        ("must_contain", "the_substring"),
        ("must_not_contain", "the_substring"),
        ("must_call_tool", "the_tool"),
        ("must_not_call_tool", "the_tool"),
    ],
)
def test_all_assertion_clauses_render(must_field, value):
    spec = TestSpec(
        strategy="assertion",
        rationale_summary="r",
        assertion=AssertionSpec(**{must_field: [value]}),
    )
    case = _assertion_case("tc-1", spec=spec)
    source = render_pytest_suite("exp-1", [case])
    assert value in source
