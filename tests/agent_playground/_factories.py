"""Shared fixtures + helpers for ``tests/agent_playground/``.

Centralizes the ``_assertion_row`` / ``_judge_row`` / ``_persona_row``
factories so the CLI and server-side router tests stay in lockstep
with the store-level tests.
"""

from __future__ import annotations

from mlflow.agent_playground.test_cases import store
from mlflow.agent_playground.test_cases.entities import (
    AssertionExpectations,
    JudgeExpectations,
    PersonaSpec,
    TestCaseRow,
)


def make_assertion_row(**overrides) -> TestCaseRow:
    """Build a baseline assertion-strategy test case for tests.

    Defaults to ``must_contain=["docs"]`` + ``must_call_tool=["search_docs"]``
    so both substring-style and tool-call clauses are exercised; pass
    ``expectations=AssertionExpectations(...)`` to override.
    """
    defaults = {
        "test_case_id": store.new_test_case_id(),
        "expectations": AssertionExpectations(
            must_contain=["docs"], must_call_tool=["search_docs"]
        ),
        "rationale_summary": "agent must cite docs",
    }
    defaults.update(overrides)
    return TestCaseRow(**defaults)


def make_judge_row(**overrides) -> TestCaseRow:
    """Build a baseline judge-strategy test case for tests."""
    defaults = {
        "test_case_id": store.new_test_case_id(),
        "expectations": JudgeExpectations(
            instructions="response is friendly", expected_response="hi friend"
        ),
        "rationale_summary": "agent should sound friendlier",
    }
    defaults.update(overrides)
    return TestCaseRow(**defaults)


def make_persona_row(**overrides) -> TestCaseRow:
    """Build a baseline persona-driven (multi-turn) test case for tests."""
    defaults = {
        "test_case_id": store.new_test_case_id(),
        "expectations": AssertionExpectations(
            must_contain=["docs"], must_call_tool=["search_docs"]
        ),
        "rationale_summary": "agent must cite docs",
        "persona": PersonaSpec(
            goal="learn about logging",
            persona="terse Python developer",
            simulation_guidelines=["ask one question at a time"],
        ),
        "max_turns": 3,
    }
    defaults.update(overrides)
    return TestCaseRow(**defaults)
