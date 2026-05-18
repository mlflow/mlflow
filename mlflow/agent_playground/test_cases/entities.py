"""Schema entities for the agent_playground test-case slice.

Pydantic models for the data plane: the per-row ``expectations``
discriminated union, the persona that lives on ``inputs.persona`` of a
row, the verdict shape the runner emits, the dedup-verdict shape the
test-gen worker hands the coder, and the feedback-anchor the prompt
builders read.

Every other module in this slice imports types from here. Nothing in
this module imports from MLflow beyond ``pydantic``; entities are
deliberately I/O-free.

Shape decisions (see ``mlflow/internal:docs/projects/agent-playground/
test-case-slice-design.md``):

- ``expectations`` is a discriminated union on ``kind`` per the
  "Expectations shape" section of the design doc, not a flat-with-Nones
  blob containing a strategy + payloads.
- The fixed prompt template for the judge strategy lives in the
  judge-strategy evaluator (stack 9), not on the row. The per-row
  field is ``instructions``, matching ``mlflow.genai.judges.make_judge``.
- ``PersonaSpec`` lives on ``inputs.persona`` of the row, sibling of
  ``inputs.messages``, matching ``ConversationSimulator``'s
  ``_EXPECTED_TEST_CASE_KEYS`` shape.
- Job lifecycle reuses ``mlflow.entities._job_status.JobStatus`` and
  ``mlflow.server.job_api.Job``; the test-gen failure taxonomy lives
  in ``JobEntity.status_details["failure_kind"]``, not in a wrapper
  pydantic type.
- ``AssistantMessageAnchor`` carries a ``kind`` discriminator so the
  alias ``TraceAnchor`` can widen to additional variants (tool_call,
  span_latency, ...) without an entity break.
- All models set ``extra="forbid"`` and ``frozen=True`` so typo'd
  fields fail loudly (catches malformed LLM-generated specs) and
  post-construction mutation can't bypass cross-field invariants.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Keys reserved by ``ConversationSimulator`` for injecting conversation
# history and session id. Mirrors ``_RESERVED_CONTEXT_KEYS`` in
# ``mlflow/genai/simulators/simulator.py``; the test suite asserts the
# two sets stay in lockstep so drift is caught in CI rather than at
# runtime. We duplicate rather than import to keep ``entities.py``
# free of ``mlflow.*`` imports beyond ``pydantic``.
_SIMULATOR_RESERVED_CONTEXT_KEYS = frozenset({"input", "messages", "mlflow_session_id"})

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Feedback anchor (consumed by prompt builders to render the failing context)
# ---------------------------------------------------------------------------


class AssistantMessageAnchor(BaseModel):
    """Substring inside an assistant message that anchors a feedback.

    The v1 ``TraceAnchor`` variant. Stored JSON-stringified inside an
    MLflow ``Assessment``'s ``metadata.anchor`` field. Future variants
    (tool_call, span_latency, ...) can ship under the same ``kind``
    discriminator without breaking the entity contract; v1 ships only
    this one.

    ``start`` / ``end`` are Python code-point offsets into the
    assistant message text, not DOM offsets or UTF-16 code units, so
    the anchor survives re-renders. Callers serializing offsets from a
    UI selection range (which is UTF-16-based in DOM APIs) must
    normalize before constructing the anchor. ``prefix`` and ``suffix``
    are a few characters around the selection used to re-resolve the
    range if the rendered text shifts.
    """

    model_config = _STRICT_MODEL_CONFIG

    kind: Literal["assistant_message"] = "assistant_message"
    message_id: str = Field(min_length=1)
    trace_id: str | None = None
    start: int
    end: int
    selected_text: str
    prefix: str
    suffix: str

    @model_validator(mode="after")
    def _validate_anchor_range(self) -> AssistantMessageAnchor:
        if self.start < 0:
            raise ValueError(f"start must be non-negative, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        if len(self.selected_text) != self.end - self.start:
            raise ValueError(
                f"selected_text length ({len(self.selected_text)}) "
                f"must equal end - start ({self.end - self.start})"
            )
        return self


# Widened to a union when additional variants ship (tool_call anchor,
# span_latency anchor, ...). v1 ships only the assistant-message
# variant so the alias collapses to a single type. Consumers that
# round-trip via ``TypeAdapter(TraceAnchor)`` must always include
# ``kind`` in dict payloads so the v1->v2 widening is a no-op at the
# wire level. ``AssistantMessageAnchor.model_dump()`` emits ``kind``
# by default.
TraceAnchor: TypeAlias = AssistantMessageAnchor


# ---------------------------------------------------------------------------
# Expectations (discriminated union on ``kind``; per-row evaluation data)
# ---------------------------------------------------------------------------


def _validate_non_empty_string_list(values: list[str], field_name: str) -> None:
    for value in values:
        if not value.strip():
            raise ValueError(
                f"{field_name} entries must be non-empty and non-whitespace, got {value!r}"
            )


class AssertionExpectations(BaseModel):
    """Deterministic substring + tool-call clauses for the assertion strategy.

    Stored as the ``expectations`` field on a test-case row. The
    ``kind`` discriminator is what the runner dispatches on.

    All four lists reject empty / whitespace-only items: an empty
    ``must_contain`` needle is a vacuous pass (``"" in any_str`` is
    always true), and an empty ``must_not_contain`` needle is a
    guaranteed fail. Catching at the entity layer prevents malformed
    LLM-generated specs from running as no-op checks. Contradictory
    pairs (the same string in ``must_contain`` and ``must_not_contain``,
    or in ``must_call_tool`` and ``must_not_call_tool``) are likewise
    rejected because they would guarantee failure on every run.
    """

    model_config = _STRICT_MODEL_CONFIG

    kind: Literal["assertion"] = "assertion"
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    must_call_tool: list[str] = Field(default_factory=list)
    must_not_call_tool: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reject_blank_clauses(self) -> AssertionExpectations:
        _validate_non_empty_string_list(self.must_contain, "AssertionExpectations.must_contain")
        _validate_non_empty_string_list(
            self.must_not_contain, "AssertionExpectations.must_not_contain"
        )
        _validate_non_empty_string_list(self.must_call_tool, "AssertionExpectations.must_call_tool")
        _validate_non_empty_string_list(
            self.must_not_call_tool, "AssertionExpectations.must_not_call_tool"
        )
        return self

    @model_validator(mode="after")
    def _reject_contradictory_clauses(self) -> AssertionExpectations:
        if contains_conflict := sorted(set(self.must_contain) & set(self.must_not_contain)):
            raise ValueError(
                "AssertionExpectations.must_contain and must_not_contain overlap on "
                f"{contains_conflict}; every run would fail"
            )
        if tool_conflict := sorted(set(self.must_call_tool) & set(self.must_not_call_tool)):
            raise ValueError(
                "AssertionExpectations.must_call_tool and must_not_call_tool overlap on "
                f"{tool_conflict}; every run would fail"
            )
        return self


class JudgeExpectations(BaseModel):
    """LLM-judge instructions plus optional reference, for the judge strategy.

    Stored as the ``expectations`` field on a test-case row. The
    per-row ``instructions`` string is the parameter substituted into
    the fixed judge prompt template; that template lives in the
    judge-strategy evaluator (stack 9), mirroring how ``Correctness``
    keeps ``CORRECTNESS_PROMPT_INSTRUCTIONS`` outside ``expectations``.
    The field name ``instructions`` matches the existing public
    ``mlflow.genai.judges.make_judge(instructions=...)`` API so the
    same word carries the same meaning across the genai/judges surface.
    ``expected_response`` is optional reference output the judge can
    use as a comparator.
    """

    model_config = _STRICT_MODEL_CONFIG

    kind: Literal["judge"] = "judge"
    instructions: str = Field(min_length=1)
    expected_response: str | None = None

    @model_validator(mode="after")
    def _reject_blank_instructions(self) -> JudgeExpectations:
        if not self.instructions.strip():
            raise ValueError("JudgeExpectations.instructions must be non-whitespace")
        return self


Expectations: TypeAlias = Annotated[
    AssertionExpectations | JudgeExpectations,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Persona (simulator input; lives on ``inputs.persona`` of the row)
# ---------------------------------------------------------------------------


class PersonaSpec(BaseModel):
    """Strict subset of ``ConversationSimulator``'s test-case dict.

    Stored on ``inputs.persona`` of the test-case row (sibling of
    ``inputs.messages``), matching the simulator's own
    ``_EXPECTED_TEST_CASE_KEYS`` shape at
    ``mlflow/genai/simulators/simulator.py:52``. The runner hands
    ``PersonaSpec.model_dump(exclude_none=True)`` straight to
    ``ConversationSimulator(test_cases=[...])`` after normalizing the
    ``simulation_guidelines`` value to a ``list[str]``.

    ``simulation_guidelines`` is stored as ``list[str]`` even though
    the simulator also accepts a bare ``str``: list-shape gives the
    UI a stable edit surface (per-guideline rows, drag-reorder) and
    makes diffing across test-case edits trivial. The drift-guard
    test pins the value-shape compatibility against the simulator's
    declared parameter type so a future narrowing upstream surfaces in
    CI.

    The simulator's own ``expectations`` field is deliberately not
    represented here because the row-level ``expectations``
    discriminated union (``AssertionExpectations`` /
    ``JudgeExpectations``) serves the same scoring-data role for our
    runner.
    """

    model_config = _STRICT_MODEL_CONFIG

    goal: str = Field(min_length=1)
    persona: str | None = None
    simulation_guidelines: list[str] | None = None
    context: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _reject_blank_goal(self) -> PersonaSpec:
        if not self.goal.strip():
            raise ValueError("PersonaSpec.goal must be non-whitespace")
        return self

    @model_validator(mode="after")
    def _reject_simulator_reserved_context_keys(self) -> PersonaSpec:
        if self.context is None:
            return self
        if conflicts := sorted(set(self.context) & _SIMULATOR_RESERVED_CONTEXT_KEYS):
            raise ValueError(
                f"PersonaSpec.context keys {conflicts} conflict with simulator-reserved "
                f"keys ({sorted(_SIMULATOR_RESERVED_CONTEXT_KEYS)}). Rename to avoid the clash."
            )
        return self


# ---------------------------------------------------------------------------
# Runner output
# ---------------------------------------------------------------------------


VerdictOutcome = Literal["pass", "fail", "error"]


class Verdict(BaseModel):
    """Outcome of a single test-case run.

    Emitted per case by the runner. ``outcome`` is one of:

    - ``"pass"``: every assertion / judge check held. ``reasons`` is
      empty.
    - ``"fail"``: the agent responded but at least one check failed.
      ``reasons`` carries the failed-clause descriptions.
    - ``"error"``: execution itself failed (agent crash, timeout,
      exception in eval). ``reasons`` carries the error description.

    ``judge_rationale`` is populated by judge-strategy cases; the
    entity layer does not enforce this because cross-strategy edge
    cases (e.g. a hybrid evaluator) would surface as artificial
    validation failures.
    ``trace_ids`` carries the per-turn agent traces emitted during the
    run, tagged with ``agent_playground.test_case_id`` and
    ``agent_playground.run_id`` so the UI can link verdicts to traces.
    """

    model_config = _STRICT_MODEL_CONFIG

    test_case_id: str = Field(min_length=1)
    outcome: VerdictOutcome
    reasons: tuple[str, ...] = ()
    judge_rationale: str | None = None
    trace_ids: tuple[str, ...] = ()
    duration_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_reasons_match_outcome(self) -> Verdict:
        if self.outcome == "pass" and self.reasons:
            raise ValueError("outcome='pass' must not carry reasons")
        if self.outcome in ("fail", "error") and not self.reasons:
            raise ValueError(f"outcome={self.outcome!r} must carry at least one reason")
        return self


# ---------------------------------------------------------------------------
# Persisted row view (canonical read + write shape for the store layer)
# ---------------------------------------------------------------------------


class TestCaseRow(BaseModel):
    """Server-internal denormalized view of a single test-case row.

    Returned by :func:`mlflow.agent_playground.test_cases.store.list_cases`
    / :func:`get_case` and accepted by :func:`insert_case`. The store
    layer maps this back and forth from the underlying
    ``EvaluationDataset`` row shape:

    - ``inputs.persona`` <-> :attr:`persona`
    - ``inputs.messages`` <-> :attr:`conversation_messages`
    - ``expectations`` <-> :attr:`expectations` (discriminated union)
    - ``tags.rationale_summary`` <-> :attr:`rationale_summary`
    - ``tags.max_turns`` <-> :attr:`max_turns`
    - ``tags.source_feedback_ids`` <-> :attr:`source_feedback_ids`
      (comma-joined on the wire, attach-order list in memory)
    - ``tags.source_trace_id`` <-> :attr:`source_trace_id`
    - ``tags.source_assistant_message_id`` <-> :attr:`source_assistant_message_id`
    - ``tags.promoted`` <-> :attr:`promoted`

    The UI talks to the underlying ``DatasetRecord`` directly on the
    wire; this is a server-internal convenience for the store layer
    and its callers (runner, CRUD router, test-gen worker).

    ``__test__ = False`` prevents pytest from collecting this as a
    test class (its name starts with ``Test``).
    """

    __test__ = False

    model_config = _STRICT_MODEL_CONFIG

    test_case_id: str = Field(min_length=1)
    expectations: Expectations
    rationale_summary: str = Field(min_length=1)
    persona: PersonaSpec | None = None
    conversation_messages: list[dict[str, Any]] = Field(default_factory=list)
    max_turns: int = Field(default=5, ge=1)
    source_feedback_ids: list[str] = Field(default_factory=list)
    source_trace_id: str | None = None
    source_assistant_message_id: str | None = None
    promoted: bool = False


# ---------------------------------------------------------------------------
# Coder-mediated dedup
# ---------------------------------------------------------------------------


class DedupVerdict(BaseModel):
    """Output of the coder-mediated semantic dedup pass.

    Returned by ``CoderAdapter.run_task(prompt,
    output_schema=DedupVerdict)`` when the test-gen worker hands the
    connected coding agent the new test case's rationale summary plus
    the existing rationale summaries in the experiment. The worker
    consults this only after the hard-match pre-filter (``dedup.py``)
    returns ``HardMatchUnique``. On ``is_duplicate=True`` the worker
    appends the new feedback's id to the existing case's
    ``source_feedback_ids`` instead of inserting a new row.
    """

    model_config = _STRICT_MODEL_CONFIG

    is_duplicate: bool
    existing_test_case_id: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _duplicate_requires_existing_id(self) -> DedupVerdict:
        if self.is_duplicate and self.existing_test_case_id is None:
            raise ValueError("is_duplicate=True requires existing_test_case_id to be set")
        if not self.is_duplicate and self.existing_test_case_id is not None:
            raise ValueError("is_duplicate=False must not carry existing_test_case_id")
        return self


__all__ = [
    "AssertionExpectations",
    "AssistantMessageAnchor",
    "DedupVerdict",
    "Expectations",
    "JudgeExpectations",
    "PersonaSpec",
    "TestCaseRow",
    "TraceAnchor",
    "Verdict",
    "VerdictOutcome",
]
