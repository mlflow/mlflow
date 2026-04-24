from __future__ import annotations

from dataclasses import dataclass, field

import pydantic

from mlflow.entities.issue import Issue, IssueSeverity
from mlflow.entities.trace import Trace
from mlflow.genai.discovery.constants import RATIONALE_TRUNCATION_LIMIT


@dataclass
class _TriageResult:
    failing_traces: list[Trace]
    rationale_map: dict[str, str]
    categories_map: dict[str, list[str]]


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
        description="Indices into the input trace summary list that exemplify this issue"
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
        default="",
        description=(
            "For EACH assigned category, explain in 1-2 sentences WHY this issue "
            "belongs to that category with specific evidence from the failure. "
            "This field is REQUIRED if any categories are assigned. "
            "E.g. 'execution: The assistant claimed playback resumed when no action occurred. "
            "correctness: It provided conflicting timer states in adjacent responses.'"
        ),
    )
