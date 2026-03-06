from __future__ import annotations

from dataclasses import dataclass

import pydantic

from mlflow.genai.discovery.constants import SeverityLevel


@dataclass
class Issue:
    issue_id: str
    run_id: str
    name: str
    description: str
    root_cause: str
    example_trace_ids: list[str]
    frequency: float
    severity: SeverityLevel
    status: str = "open"
    created_at: str = ""


@dataclass
class DiscoverIssuesResult:
    issues: list[Issue]
    triage_run_id: str
    summary: str
    total_traces_analyzed: int


@dataclass
class _ConversationAnalysis:
    surface: str
    root_cause: str
    affected_trace_ids: list[str]
    execution_path: str = ""


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
    severity: SeverityLevel = pydantic.Field(
        description=(
            "Severity of this issue. "
            "not_an_issue=not a real issue, low=minor issue, "
            "medium=moderate issue, high=critical issue"
        ),
    )
