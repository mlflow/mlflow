"""Prompt builders for the agent_playground test-case lifecycle.

Two callers, two prompts:

1. ``build_test_gen_prompt`` is invoked by the async test-gen job worker
   (later stack). The connected coding agent receives this prompt plus a
   ``GeneratedTestCase`` JSON schema and produces a regression test case
   from the user's feedback.
2. ``build_fix_prompt`` is invoked by the fix-prompt REST endpoint
   (later stack). The result is what the playground UI copies to the
   clipboard for the user to paste into their terminal ``claude``
   session; the goal is for Claude to edit the agent until
   ``mlflow agent test run --test-case <id>`` exits 0.

Both functions are I/O-aware (they load assessments / traces / test
cases from MLflow) but return only strings; rendering is deterministic
given the inputs, so most tests mock the IO layer and verify the
substring content.

Trust boundary: user-supplied feedback rationale, anchored text, and
conversation content flow into the rendered prompt verbatim. The v1
playground is a single-developer tool (the developer's own feedback
goes to the developer's own coding agent), so prompt-injection
mitigations are deferred; revisit if the surface ever exposes
cross-user feedback.
"""

from __future__ import annotations

import json
from typing import Any

import pydantic

from mlflow.agent_playground.test_cases import store
from mlflow.agent_playground.test_cases.entities import (
    AssertionExpectations,
    AssistantMessageAnchor,
    JudgeExpectations,
    TestCaseRow,
)
from mlflow.entities import Assessment, Trace
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST
from mlflow.telemetry.events import AgentPlaygroundBuildPromptFromFeedbackEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracing.assessment import get_assessment
from mlflow.tracing.client import TracingClient
from mlflow.tracing.utils.truncation import _get_text_content_from_message, _try_extract_messages


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
    except (json.JSONDecodeError, TypeError):
        # ``json.loads`` raises ``JSONDecodeError`` on malformed strings
        # and ``TypeError`` on non-string inputs (e.g., a metadata dict
        # constructed in-Python with a non-str ``anchor`` value that
        # bypassed the proto-side str coercion).
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return AssistantMessageAnchor(**payload)
    except (pydantic.ValidationError, TypeError):
        return None


def _load_assessment_or_raise(trace_id: str, assessment_id: str) -> Assessment:
    try:
        return get_assessment(trace_id=trace_id, assessment_id=assessment_id)
    except MlflowException:
        raise
    except Exception as exc:
        raise MlflowException(
            f"Failed to load assessment {assessment_id!r} on trace {trace_id!r}: {exc}",
            error_code=INTERNAL_ERROR,
        ) from exc


def _load_trace_or_raise(trace_id: str) -> Trace:
    try:
        return TracingClient().get_trace(trace_id)
    except MlflowException:
        raise
    except Exception as exc:
        raise MlflowException(
            f"Failed to load trace {trace_id!r}: {exc}",
            error_code=INTERNAL_ERROR,
        ) from exc


def _extract_messages_from_trace(trace: Trace) -> list[dict[str, Any]]:
    """Pull the conversation messages out of a trace's root span.

    Delegates to the shared ``_try_extract_messages`` helper in
    ``mlflow.tracing.utils.truncation``, which dispatches across the
    OpenAI ``messages`` / ``input`` / ``choices`` / ``output`` shapes
    plus the nested ``request`` key. Returns an empty list when no
    well-formed message list is recoverable; the caller's prompt
    degrades to "no conversation context."
    """
    root = trace.data._get_root_span()
    if root is None or not isinstance(root.inputs, dict):
        return []
    messages = _try_extract_messages(root.inputs)
    return messages or []


def _format_messages(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "(no prior conversation context recorded on the trace)"
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        # ``_get_text_content_from_message`` flattens multimodal
        # ``content: list[dict]`` shapes (e.g., text + image parts)
        # into the first text-part string; inlining the f-string
        # would render the Python ``repr`` of the list.
        content = _get_text_content_from_message(msg)
        lines.append(f"  {role}: {content}")
    return "\n".join(lines)


_TEST_GEN_PROMPT_TEMPLATE = """\
You will generate a regression test case for an MLflow Agent Playground
feedback. The user flagged an assistant response as wrong; your job is
to produce a ``GeneratedTestCase`` JSON object that captures the
contract the agent should satisfy.

# User feedback
{rationale}

# Anchored substring (what the user highlighted in the response)
{anchor_text}

# Conversation that produced the failing response
{conversation}

# Your task

1. Read the agent's source code in this repo so you understand its
   tools, conventions, and the contract the failing response violates.

2. Emit a single JSON object matching the ``GeneratedTestCase`` schema
   you will be given. The schema's ``expectations`` field is a
   discriminated union on ``kind``: use ``kind="assertion"`` for
   concrete claims like "must cite docs" or "must call tool X"; use
   ``kind="judge"`` (with an ``instructions`` string) for subjective
   qualities like "should sound friendlier".

3. If the conversation has more than one user turn, populate the
   ``persona`` field (matches
   ``mlflow.genai.simulators.ConversationSimulator``'s test-case dict
   shape: goal + persona + simulation_guidelines + context). The
   runner uses the simulator to drive the user side at run time; the
   persona does NOT contain pre-baked user messages.

Do not include any prose outside the JSON object.
"""


def build_test_gen_prompt(trace_id: str, assessment_id: str) -> str:
    """Render the test-gen prompt for the connected coding agent.

    No ``@record_usage_event`` decorator: this function is only called
    from inside the test-gen job worker, which already emits
    ``AgentPlaygroundSubmitTestGenJobEvent`` and
    ``AgentPlaygroundCompleteTestGenJobEvent`` around the whole job
    lifecycle. ``build_fix_prompt`` is the UI-copy endpoint with no
    surrounding job, so it owns its own telemetry.

    Args:
        trace_id: Failing trace the feedback is anchored to.
        assessment_id: Saved feedback assessment id.

    Returns:
        Prompt string suitable for ``CoderAdapter.run_task(prompt,
        output_schema=GeneratedTestCase)`` (the adapter wraps the
        JSON-schema instruction itself per the proposal at
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

# Expectations on the final assistant response
{expectations_block}

# Your task

Edit the agent's source code so the test passes. Verify with:

  mlflow agent test run --test-case {test_case_id}

The command must exit 0 before you claim the fix is done. Iterate as
needed: read the failure output, adjust the agent, run the test again.
Commit your changes only once the test passes.
"""


def _format_persona_block(case: TestCaseRow) -> str:
    persona = case.persona
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
    if persona.context:
        # Simulator-injected variables; surface to the coding agent so
        # it can reason about which values the test will run under.
        lines.append(f"  context: {json.dumps(persona.context, sort_keys=True)}")
    lines.append(f"  max_turns: {case.max_turns}")
    return "\n".join(lines)


def _format_expectations_block(case: TestCaseRow) -> str:
    # ``case.expectations`` is the ``Expectations`` discriminated union;
    # dispatch on the variant type so a future widening of the union
    # (a third ``kind="..."`` variant) falls through to the explicit
    # raise below rather than silently mis-rendering as an assertion.
    expectations = case.expectations
    if isinstance(expectations, JudgeExpectations):
        lines = [f"  instructions: {expectations.instructions}"]
        if expectations.expected_response:
            lines.append(f"  expected:     {expectations.expected_response}")
        return "\n".join(lines)
    if isinstance(expectations, AssertionExpectations):
        lines = []
        if expectations.must_contain:
            lines.append(f"  must_contain:       {expectations.must_contain}")
        if expectations.must_not_contain:
            lines.append(f"  must_not_contain:   {expectations.must_not_contain}")
        if expectations.must_call_tool:
            lines.append(f"  must_call_tool:     {expectations.must_call_tool}")
        if expectations.must_not_call_tool:
            lines.append(f"  must_not_call_tool: {expectations.must_not_call_tool}")
        if not lines:
            lines.append("  (no clauses; any response passes)")
        return "\n".join(lines)
    raise MlflowException(
        f"Unhandled expectations kind {expectations.kind!r} in fix-prompt renderer",
        error_code=INTERNAL_ERROR,
    )


@record_usage_event(AgentPlaygroundBuildPromptFromFeedbackEvent)
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
        rationale_summary=case.rationale_summary,
        test_case_id=test_case_id,
        persona_block=_format_persona_block(case),
        expectations_block=_format_expectations_block(case),
    )


__all__ = [
    "build_fix_prompt",
    "build_test_gen_prompt",
]
