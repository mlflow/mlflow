"""Pytest export generator for agent_playground regression suites.

Renders a self-contained pytest file the user commits to CI as a
regression gate against their agent. The file POSTs each test case's
conversation messages to the agent's invocations endpoint
(``MLFLOW_AGENT_URL``, default ``http://localhost:8000/invocations``)
and runs the deterministic assertion checks defined on the case.

v1 scope: assertion-strategy + single-turn cases. Multi-turn (persona)
and judge-strategy cases are excluded with a count in the file header
because:

- Multi-turn cases need the ``mlflow.genai.simulators`` conversation
  simulator to drive user turns, which depends on a configured judge
  LLM. Not portable to a hermetic CI gate.
- Judge-strategy cases call an LLM to score the response. Also not
  portable to hermetic CI.

Users who want those cases in CI can run ``mlflow agent test run``
against a configured MLflow server instead.

Response shape: the generated file assumes a chat-completion response
(``{"choices": [{"message": {"content": "...", "tool_calls": [...]}}]}``).
Users whose agents return a different shape can edit the
``_extract_response_text`` / ``_extract_tool_call_names`` helpers at
the top of the generated file.
"""

from __future__ import annotations

import pprint
from dataclasses import dataclass
from datetime import datetime, timezone

from mlflow.agent_playground.test_cases.entities import TestCaseRow


@dataclass(frozen=True)
class _ExportablePartition:
    exportable: list[TestCaseRow]
    skipped_judge: int
    skipped_persona: int
    skipped_no_messages: int


def _partition(cases: list[TestCaseRow]) -> _ExportablePartition:
    exportable: list[TestCaseRow] = []
    skipped_judge = 0
    skipped_persona = 0
    skipped_no_messages = 0
    for case in cases:
        if case.spec.strategy == "judge":
            skipped_judge += 1
            continue
        if case.spec.persona is not None:
            skipped_persona += 1
            continue
        if not case.conversation_messages:
            skipped_no_messages += 1
            continue
        exportable.append(case)
    return _ExportablePartition(
        exportable=exportable,
        skipped_judge=skipped_judge,
        skipped_persona=skipped_persona,
        skipped_no_messages=skipped_no_messages,
    )


_FILE_HEADER_TEMPLATE = '''"""Auto-generated agent_playground regression suite.

Experiment: {experiment_id}
Generated: {generated_at}
Cases included: {included}
Cases excluded: {excluded_total}
  (judge: {skipped_judge}, persona: {skipped_persona}, no-conversation: {skipped_no_messages})

DO NOT EDIT MANUALLY. Regenerate with::

    mlflow agent test export --experiment-id {experiment_id} --output <path>

The suite POSTs each case's conversation to the agent at
``MLFLOW_AGENT_URL`` (default ``http://localhost:8000/invocations``)
and runs deterministic substring / tool-call assertions on the
response. Override ``MLFLOW_AGENT_URL`` in CI to point at the
deployment under test.
"""

from __future__ import annotations

import os

import pytest
import requests

AGENT_URL = os.environ.get("MLFLOW_AGENT_URL", "http://localhost:8000/invocations")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("MLFLOW_AGENT_TIMEOUT_SECONDS", "30"))


def _extract_response_text(response: dict) -> str:
    """Pull the assistant text out of a chat-completion response.

    Customize if your agent returns a different shape.
    """
    return response["choices"][0]["message"]["content"] or ""


def _extract_tool_call_names(response: dict) -> list[str]:
    """Pull tool-call function names from a chat-completion response.

    Returns an empty list if the response has no tool calls.
    Customize if your agent returns a different shape.
    """
    message = response["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    return [call["function"]["name"] for call in tool_calls]
'''


_TEST_BODY = """
def _run_case(case: dict) -> None:
    resp = requests.post(
        AGENT_URL,
        json={"messages": case["messages"]},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    body = resp.json()

    text = _extract_response_text(body)
    tool_calls = _extract_tool_call_names(body)

    reasons: list[str] = []
    for needle in case["must_contain"]:
        if needle not in text:
            reasons.append(f"must_contain: {needle!r} not in response")
    for needle in case["must_not_contain"]:
        if needle in text:
            reasons.append(f"must_not_contain: {needle!r} present in response")
    called = set(tool_calls)
    for tool in case["must_call_tool"]:
        if tool not in called:
            reasons.append(f"must_call_tool: {tool!r} not invoked")
    for tool in case["must_not_call_tool"]:
        if tool in called:
            reasons.append(f"must_not_call_tool: {tool!r} was invoked")

    assert not reasons, "\\n".join(reasons)


@pytest.mark.parametrize("case", _CASES, ids=[c["test_case_id"] for c in _CASES])
def test_agent_playground_regression(case: dict) -> None:
    _run_case(case)
"""


_EMPTY_SUITE_BODY = """\
import pytest

pytest.skip(
    "No exportable cases in this experiment. Add assertion-strategy "
    "single-turn cases (judge-strategy and multi-turn cases are not "
    "exportable to a hermetic CI gate) and re-run "
    "`mlflow agent test export`.",
    allow_module_level=True,
)
"""


def _case_payload(case: TestCaseRow) -> dict[str, object]:
    assertion = case.spec.assertion
    return {
        "test_case_id": case.test_case_id,
        "rationale_summary": case.spec.rationale_summary,
        "messages": case.conversation_messages,
        "must_contain": list(assertion.must_contain) if assertion else [],
        "must_not_contain": list(assertion.must_not_contain) if assertion else [],
        "must_call_tool": list(assertion.must_call_tool) if assertion else [],
        "must_not_call_tool": list(assertion.must_not_call_tool) if assertion else [],
    }


def render_pytest_suite(experiment_id: str, cases: list[TestCaseRow]) -> str:
    """Render a self-contained pytest file string for ``cases``.

    Args:
        experiment_id: Owning experiment, recorded in the file header.
        cases: Every case in the regression suite. Non-exportable cases
            are filtered out and counted in the file header.

    Returns:
        Python source as a string. Always valid Python regardless of
        whether ``cases`` is empty (an empty suite renders a file with
        no parametrized test cases).
    """
    partition = _partition(cases)
    header = _FILE_HEADER_TEMPLATE.format(
        experiment_id=experiment_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        included=len(partition.exportable),
        excluded_total=(
            partition.skipped_judge + partition.skipped_persona + partition.skipped_no_messages
        ),
        skipped_judge=partition.skipped_judge,
        skipped_persona=partition.skipped_persona,
        skipped_no_messages=partition.skipped_no_messages,
    )

    payloads = [_case_payload(case) for case in partition.exportable]
    if not payloads:
        # Empty suite: skip at module level so the file is collectable
        # by pytest without producing a misleading ``[NOTSET]``
        # parametrized test. Users see one clean ``SKIPPED`` line.
        return header + "\n\n" + _EMPTY_SUITE_BODY

    # ``pprint.pformat`` emits valid Python literals (``None`` / ``True``
    # / ``False`` for nulls/bools) so the rendered file is executable.
    # ``json.dumps`` would emit ``null`` / ``true`` / ``false`` which
    # ``NameError`` at module load. ``sort_dicts=False`` preserves the
    # author's field order.
    cases_literal = "_CASES: list[dict] = " + pprint.pformat(
        payloads, indent=4, width=100, sort_dicts=False
    )

    return header + "\n\n" + cases_literal + "\n" + _TEST_BODY
