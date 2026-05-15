"""Prompt builders for the agent_playground test-case lifecycle.

Two callers, two prompts:

1. ``build_test_gen_prompt`` is invoked by the async test-gen job worker
   (later stack). The connected coding agent receives this prompt plus a
   ``TestSpec`` JSON schema and produces a regression test case from the
   user's feedback.
2. ``build_fix_prompt`` is invoked by the fix-prompt REST endpoint
   (later stack). The result is what the playground UI copies to the
   clipboard for the user to paste into their terminal ``claude``
   session; the goal is for Claude to edit the agent until
   ``mlflow agent test run --test-case <id>`` exits 0.

Both functions are I/O-aware (they load assessments / traces / test
cases from MLflow) but return only strings; rendering is deterministic
given the inputs, so most tests mock the IO layer and verify the
substring content.
"""

from __future__ import annotations

import json
from typing import Any

from mlflow.agent_playground.test_cases import store
from mlflow.agent_playground.test_cases.entities import (
    AssistantMessageAnchor,
    TestCaseRow,
)
from mlflow.entities import Assessment, Trace
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.tracing.assessment import get_assessment
from mlflow.tracking.client import MlflowClient


def _parse_anchor(metadata: dict[str, str] | None) -> AssistantMessageAnchor | None:
    """Extract the assistant-message anchor from an assessment's metadata.

    The widget stores the anchor as a JSON string under
    ``metadata.anchor``. Returns ``None`` when no anchor is present or
    when the stored value is malformed (the prompt degrades gracefully
    in that case by omitting the anchored substring).
    """
    if not metadata:
        return None
    raw = metadata.get("anchor")
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return AssistantMessageAnchor(**payload)
    except Exception:
        return None


def _extract_messages_from_trace(trace: Trace) -> list[dict[str, Any]]:
    """Pull the conversation messages out of a trace's root span.

    The agent's ``@invoke`` contract returns a response payload whose
    span inputs are the conversation prefix. If the inputs aren't a
    well-formed message list, returns an empty list and the caller's
    prompt degrades to "no conversation context."
    """
    spans = getattr(trace.data, "spans", []) or []
    if not spans:
        return []
    root = spans[0]
    inputs = getattr(root, "inputs", None)
    messages = inputs.get("messages") or inputs.get("input") if isinstance(inputs, dict) else inputs
    if isinstance(messages, list):
        return [m for m in messages if isinstance(m, dict)]
    return []


def _format_messages(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "(no prior conversation context recorded on the trace)"
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"  {role}: {content}")
    return "\n".join(lines)


_TEST_GEN_PROMPT_TEMPLATE = """\
You will generate a regression test case for an MLflow Agent Playground
feedback. The user flagged an assistant response as wrong; your job is
to produce a TestSpec JSON that captures the contract the agent should
satisfy.

# User feedback
{rationale}

# Anchored substring (what the user highlighted in the response)
{anchor_text}

# Conversation that produced the failing response
{conversation}

# Your task

1. Read the agent's source code in this repo so you understand its
   tools, conventions, and the contract the failing response violates.

2. Emit a single JSON object matching the TestSpec schema you will be
   given. Pick `strategy="assertion"` for concrete claims like "must
   cite docs" or "must call tool X"; pick `strategy="judge"` for
   subjective qualities like "should sound friendlier".

3. If the conversation has more than one user turn, include a `persona`
   block (matches `mlflow.genai.simulators.ConversationSimulator`'s
   test-case dict shape: goal + persona + simulation_guidelines +
   context). The runner uses the simulator to drive the user side at
   run time; the persona does NOT contain pre-baked user messages.

Do not include any prose outside the JSON object.
"""


def build_test_gen_prompt(trace_id: str, assessment_id: str) -> str:
    """Render the test-gen prompt for the connected coding agent.

    Args:
        trace_id: Failing trace the feedback is anchored to.
        assessment_id: Saved feedback assessment id.

    Returns:
        Prompt string suitable for ``CoderAdapter.run_task(prompt,
        output_schema=TestSpec)`` (the adapter wraps the JSON-schema
        instruction itself per the proposal at
        ``mlflow/internal:docs/projects/agent-playground/coder-adapter-proposal.md``).

    Raises:
        MlflowException: When the assessment or trace cannot be loaded.
    """
    assessment = _load_assessment_or_raise(trace_id, assessment_id)
    anchor = _parse_anchor(assessment.metadata)
    anchor_text = anchor.selected_text if anchor else "(no anchored substring recorded)"

    trace = _load_trace_or_raise(trace_id)
    messages = _extract_messages_from_trace(trace)

    return _TEST_GEN_PROMPT_TEMPLATE.format(
        rationale=assessment.rationale or "(no rationale recorded)",
        anchor_text=anchor_text,
        conversation=_format_messages(messages),
    )


_FIX_PROMPT_TEMPLATE = """\
You are fixing the user's agent so it passes a new regression test
generated from their feedback.

# Original user feedback
{rationale_summary}

# Test case {test_case_id}
Persona-driven conversation that the test replays against the agent:

{persona_block}

# Assertions on the final assistant response
{assertion_block}

# Your task

Edit the agent's source code so the test passes. Verify with:

  mlflow agent test run --test-case {test_case_id}

The command must exit 0 before you claim the fix is done. Iterate as
needed: read the failure output, adjust the agent, run the test again.
Commit your changes only once the test passes.
"""


def _format_persona_block(case: TestCaseRow) -> str:
    persona = case.spec.persona
    if persona is None:
        return "  (single-turn test; no persona)"
    lines = [
        f"  goal:    {persona.goal}",
    ]
    if persona.persona:
        lines.append(f"  persona: {persona.persona}")
    if persona.simulation_guidelines:
        joined = "; ".join(persona.simulation_guidelines)
        lines.append(f"  guidelines: {joined}")
    lines.append(f"  max_turns: {case.spec.max_turns}")
    return "\n".join(lines)


def _format_assertion_block(case: TestCaseRow) -> str:
    if case.spec.strategy == "judge":
        if case.spec.judge is None:
            return "  (judge strategy with empty payload)"
        lines = [f"  criteria: {case.spec.judge.criteria}"]
        if case.spec.judge.expected_response:
            lines.append(f"  expected: {case.spec.judge.expected_response}")
        return "\n".join(lines)

    assertion = case.spec.assertion
    if assertion is None:
        return "  (assertion strategy with empty payload)"
    lines = []
    if assertion.must_contain:
        lines.append(f"  must_contain:      {assertion.must_contain}")
    if assertion.must_not_contain:
        lines.append(f"  must_not_contain:  {assertion.must_not_contain}")
    if assertion.must_call_tool:
        lines.append(f"  must_call_tool:    {assertion.must_call_tool}")
    if assertion.must_not_call_tool:
        lines.append(f"  must_not_call_tool: {assertion.must_not_call_tool}")
    if not lines:
        lines.append("  (no clauses; any response passes)")
    return "\n".join(lines)


def build_fix_prompt(experiment_id: str, test_case_id: str) -> str:
    """Render the fix prompt that the UI copies to the user's clipboard.

    Args:
        experiment_id: Owning experiment.
        test_case_id: Id of the failing test case.

    Returns:
        Prompt string the user pastes into their terminal ``claude``
        session.

    Raises:
        MlflowException: When the test case cannot be found.
    """
    case = store.get_case(experiment_id, test_case_id)
    if case is None:
        raise MlflowException(
            f"Test case {test_case_id!r} not found in experiment {experiment_id!r}",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    return _FIX_PROMPT_TEMPLATE.format(
        rationale_summary=case.spec.rationale_summary,
        test_case_id=test_case_id,
        persona_block=_format_persona_block(case),
        assertion_block=_format_assertion_block(case),
    )


def _load_assessment_or_raise(trace_id: str, assessment_id: str) -> Assessment:
    try:
        return get_assessment(trace_id=trace_id, assessment_id=assessment_id)
    except MlflowException:
        raise
    except Exception as exc:
        raise MlflowException(
            f"Failed to load assessment {assessment_id!r} on trace {trace_id!r}: {exc}"
        ) from exc


def _load_trace_or_raise(trace_id: str) -> Trace:
    try:
        return MlflowClient().get_trace(trace_id)
    except MlflowException:
        raise
    except Exception as exc:
        raise MlflowException(f"Failed to load trace {trace_id!r}: {exc}") from exc


__all__ = [
    "build_fix_prompt",
    "build_test_gen_prompt",
]
