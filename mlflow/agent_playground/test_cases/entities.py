"""Schema entities for the agent_playground test-case slice.

Pydantic models for everything in the data plane: the row shape stored in
the regression dataset, the persona profile handed to
``mlflow.genai.simulators.ConversationSimulator``, the verdict shape the
runner emits, the job-status response shape the UI polls, and the
feedback assessment anchor the prompt builder reads.

Every other module in this slice imports types from here. Nothing in
this module imports from MLflow beyond ``pydantic``; entities are
deliberately I/O-free.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

import pydantic
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Feedback anchor (consumed by prompt builders to render the failing context)
# ---------------------------------------------------------------------------


class AssistantMessageAnchor(BaseModel):
    """Logical anchor for a piece of feedback against an assistant message.

    Stored JSON-stringified inside an MLflow ``Assessment``'s
    ``metadata.anchor`` field. Matches the shape Yuki's feedback widget
    writes (POC ``feedback.tsx::AssistantMessageAnchor``).

    Character offsets are into the assistant message text, not DOM
    offsets, so the anchor survives re-renders. ``prefix`` and ``suffix``
    are a few characters around the selection used to re-resolve the
    range if the rendered text shifts.
    """

    message_id: str
    trace_id: str | None = None
    start: int
    end: int
    selected_text: str
    prefix: str
    suffix: str


# ---------------------------------------------------------------------------
# Test strategy specs
# ---------------------------------------------------------------------------


class AssertionSpec(BaseModel):
    """Deterministic check spec for the ``assertion`` test strategy."""

    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    must_call_tool: list[str] = Field(default_factory=list)
    must_not_call_tool: list[str] = Field(default_factory=list)


class JudgeSpec(BaseModel):
    """LLM-judge spec for the ``judge`` test strategy.

    The judge LLM is the connected coding agent (via Haru's
    ``CoderAdapter``) by default. ``expected_response`` is optional
    reference output the judge can use as a comparator when scoring.
    """

    criteria: str
    expected_response: str | None = None


# ---------------------------------------------------------------------------
# Persona (multi-turn simulator input)
# ---------------------------------------------------------------------------


class PersonaSpec(BaseModel):
    """Strict subset of ``ConversationSimulator``'s test case dict.

    The runner hands ``PersonaSpec.model_dump(exclude_none=True)``
    straight to ``ConversationSimulator(test_cases=[...])`` without
    translation. Field naming matches the simulator's
    ``_EXPECTED_TEST_CASE_KEYS`` (see
    ``mlflow/genai/simulators/simulator.py``).

    The simulator's ``expectations`` field is deliberately skipped to
    avoid a name collision with the row-level ``expectations`` column on
    ``EvaluationDataset``, which stores the parent ``TestSpec``.
    """

    goal: str
    persona: str | None = None
    simulation_guidelines: list[str] | None = None
    context: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Top-level test spec
# ---------------------------------------------------------------------------


TestStrategy = Literal["assertion", "judge"]


class TestSpec(BaseModel):
    """The contents of a regression test case.

    Stored inside the ``expectations`` column of the
    ``regression_suite_<exp_id>`` ``EvaluationDataset``. The runner
    reads this to know how to drive a test (single-turn vs multi-turn
    via persona) and how to score the result (assertion vs judge).

    ``max_turns`` overrides the simulator-level default per case.
    Persona-less test cases run single-turn (no simulator).
    """

    # The class name starts with "Test", which makes pytest try to
    # collect it as a test class. This flag tells pytest to skip it.
    __test__ = False

    strategy: TestStrategy
    rationale_summary: str
    max_turns: int = 5
    assertion: AssertionSpec | None = None
    judge: JudgeSpec | None = None
    persona: PersonaSpec | None = None

    @pydantic.model_validator(mode="after")
    def _strategy_matches_payload(self) -> TestSpec:
        match self.strategy:
            case "assertion" if self.assertion is None:
                raise ValueError("strategy='assertion' requires an assertion payload")
            case "judge" if self.judge is None:
                raise ValueError("strategy='judge' requires a judge payload")
        return self


# ---------------------------------------------------------------------------
# Runner outputs
# ---------------------------------------------------------------------------


class Verdict(BaseModel):
    """Outcome of a single test-case run.

    Emitted per case by the runner; aggregated into a ``RunSummary`` for
    the parent batch. ``reasons`` is empty when ``passed`` is True.
    ``judge_rationale`` is populated only for judge-strategy cases.
    ``trace_ids`` carries the per-turn agent traces emitted during the
    run, tagged with ``agent_playground.test_case_id`` and
    ``agent_playground.run_id`` so the UI can link verdicts to traces.
    """

    test_case_id: str
    passed: bool
    reasons: tuple[str, ...] = ()
    judge_rationale: str | None = None
    trace_ids: tuple[str, ...] = ()
    duration_ms: int | None = None


class RunSummary(BaseModel):
    """Aggregate summary of a batch test-suite run.

    Persisted as an MLflow ``Run`` artifact alongside the run's standard
    fields. The parent ``Run`` is tagged with the existing
    ``MLFLOW_RUN_TYPE`` tag (value ``agent_playground_regression``).
    """

    run_id: str
    pass_count: int
    fail_count: int
    error_count: int = 0
    duration_ms: int


# ---------------------------------------------------------------------------
# Async test-gen job (server-side)
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Lifecycle states for a test-gen job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


JobFailureKind = Literal[
    "timeout",
    "missing_binary",
    "not_ready",
    "schema_validation",
    "other",
]


class JobResponse(BaseModel):
    """Shape returned by ``POST /test-cases/jobs`` and the poll endpoint.

    On the trigger call, ``status`` is ``pending`` and ``test_case_id``
    is absent. On a successful poll, ``status`` is ``succeeded`` and
    ``test_case_id`` is the persisted row id (``deduped=True`` indicates
    the new feedback was attached to an existing case rather than
    creating a new row). On failure, ``status`` is ``failed`` and the
    ``failure_kind`` discriminator plus ``failure_reason`` message
    describe the cause for the UI to surface.
    """

    job_id: str
    status: JobStatus
    test_case_id: str | None = None
    deduped: bool = False
    failure_reason: str | None = None
    failure_kind: JobFailureKind | None = None


# ---------------------------------------------------------------------------
# Coder-mediated dedup
# ---------------------------------------------------------------------------


class DedupVerdict(BaseModel):
    """Output of the coder-mediated semantic dedup pass.

    Returned by ``CoderAdapter.run_task`` when the test-gen worker hands
    it the new test case plus existing rationale summaries. The worker
    consults this only after the hard-match check (``dedup.py``) returns
    ``Unique``. On ``is_duplicate=True``, the worker appends the new
    feedback's id to the existing case's ``source_feedback_ids`` tag
    instead of inserting a new row.
    """

    is_duplicate: bool
    existing_test_case_id: str | None = None
    reason: str


__all__ = [
    "AssertionSpec",
    "AssistantMessageAnchor",
    "DedupVerdict",
    "JobFailureKind",
    "JobResponse",
    "JobStatus",
    "JudgeSpec",
    "PersonaSpec",
    "RunSummary",
    "TestSpec",
    "TestStrategy",
    "Verdict",
]
