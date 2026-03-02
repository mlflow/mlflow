from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pydantic

from mlflow.genai.scorers.base import Scorer

ConfidenceLevel = Literal["definitely_no", "weak_no", "maybe", "weak_yes", "definitely_yes"]

# ---- Public dataclasses ----


@dataclass
class Issue:
    name: str
    description: str
    root_cause: str
    example_trace_ids: list[str]
    scorer: Scorer | None
    frequency: float
    confidence: ConfidenceLevel
    rationale_examples: list[str] = field(default_factory=list)


@dataclass
class DiscoverIssuesResult:
    issues: list[Issue]
    triage_run_id: str
    validation_run_id: str | None
    summary: str
    total_traces_analyzed: int


# ---- Pydantic schemas for LLM structured output ----


@dataclass
class _ConversationAnalysis:
    """Full analysis — LLM fields plus programmatically-set trace IDs."""

    surface: str
    root_cause: str
    symptoms: str
    domain: str
    affected_trace_ids: list[str]
    severity: int
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
    confidence: ConfidenceLevel = pydantic.Field(
        description=(
            "Confidence that this is a real, distinct issue. "
            "definitely_no=not a real issue, weak_no=probably not real, "
            "maybe=uncertain, weak_yes=probably real, definitely_yes=certainly real"
        ),
    )
