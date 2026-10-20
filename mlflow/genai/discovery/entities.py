from __future__ import annotations

from dataclasses import dataclass, field

import pydantic

from mlflow.entities.issue import Issue, IssueSeverity
from mlflow.entities.trace import Trace
from mlflow.genai.discovery.constants import RATIONALE_TRUNCATION_LIMIT


@dataclass
class _TriageResult:
    """
    The outcome of scoring a sample, with the traces no judgment was reached on.

    A trace that produced no usable verdict is not a passing trace. Keeping the three
    outcomes apart is what stops a run whose judge failed everywhere from reporting a clean
    scan: ``failing_traces`` is what to analyze, while the other two say how much of the
    sample the run is entitled to draw conclusions from.

    Args:
        failing_traces: Traces at least one scorer marked failing.
        rationale_map: Trace ID to the combined rationale of its failing scorers.
        categories_map: Trace ID to the category tags triage assigned.
        scorer_error_trace_ids: Traces whose scorer itself errored. The scorer failed, not
            the trace, so these are reported as a scorer problem and excluded from
            trace-quality classification.
        unclassified_trace_ids: Traces a scorer returned no value for, so neither pass nor
            fail was established.
    """

    failing_traces: list[Trace]
    rationale_map: dict[str, str]
    categories_map: dict[str, list[str]]
    scorer_error_trace_ids: list[str] = field(default_factory=list)
    unclassified_trace_ids: list[str] = field(default_factory=list)
    scorer_errors: dict[str, str] = field(default_factory=dict)


@dataclass
class DiscoverIssuesResult:
    issues: list[Issue]
    triage_run_id: str
    summary: str
    total_traces_analyzed: int
    total_cost_usd: float | None = None


@dataclass
class _ConversationAnalysis:
    """Per-session analysis built from triage rationales and span errors.

    Args:
        full_rationale: Combined rationale from all failing traces in this session.
        affected_trace_ids: Trace IDs of failing traces in this session.
        execution_path: Compact path of sub-agents/tools called (e.g.
            ``"ask_sports > get_scores, web_search"``).
        categories: Category tags extracted from triage rationales (e.g. ["hallucination"]).
    """

    full_rationale: str
    affected_trace_ids: list[str]
    execution_path: str = ""
    categories: list[str] = field(default_factory=list)

    @property
    def rationale_summary(self) -> str:
        return self.full_rationale[:RATIONALE_TRUNCATION_LIMIT]


class _IdentifiedIssue(pydantic.BaseModel):
    name: str = pydantic.Field(
        description=(
            "Title prefixed with 'Issue: ' followed by a short readable description (3-8 words), "
            "e.g. 'Issue: Media control commands ignored'"
        )
    )
    description: str = pydantic.Field(description="What the issue is")
    root_cause: str = pydantic.Field(description="Why this issue occurs")
    example_indices: list[int] = pydantic.Field(
        description="Indices into the input trace summary list that exemplify this issue",
    )
    severity: IssueSeverity = pydantic.Field(
        description=(
            "Severity of this issue. "
            "not_an_issue=not a real issue, low=minor issue, "
            "medium=moderate issue, high=critical issue"
        ),
    )
    categories: list[str] = pydantic.Field(
        description=(
            "Categories of this issue. Each category MUST be substantiated by "
            "concrete evidence in the description and root cause."
        ),
    )
    category_rationale: str = pydantic.Field(
        description=(
            "For EACH assigned category, explain in 1-2 sentences WHY this issue "
            "belongs to that category with specific evidence from the failure. "
            "Return an empty string if no categories are assigned. "
            "E.g. 'execution: The assistant claimed playback resumed when no action occurred. "
            "correctness: It provided conflicting timer states in adjacent responses.'"
        ),
    )
