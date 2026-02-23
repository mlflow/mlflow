from __future__ import annotations

from dataclasses import dataclass, field

import pydantic

from mlflow.genai.scorers.base import Scorer

# ---- Public dataclasses ----


@dataclass
class Issue:
    name: str
    description: str
    root_cause: str
    example_trace_ids: list[str]
    scorer: Scorer | None
    frequency: float
    confidence: int
    rationale_examples: list[str] = field(default_factory=list)


@dataclass
class DiscoverIssuesResult:
    issues: list[Issue]
    triage_run_id: str
    validation_run_id: str | None
    summary: str
    total_traces_analyzed: int


# ---- Pydantic schemas for LLM structured output ----


class _ConversationAnalysisLLMResult(pydantic.BaseModel):
    """Schema the LLM fills in for deep analysis."""

    surface: str = pydantic.Field(
        description=(
            "Where to intervene: primary function + data/object + interaction mechanism. "
            "5-12 word noun phrase, no failure language."
        )
    )
    root_cause: str = pydantic.Field(
        description=(
            "Technical root cause of the triage-identified failure. "
            "3-5 short declarative sentences grounded in span evidence. "
            "No speculation. 40-120 words."
        )
    )
    symptoms: str = pydantic.Field(
        description=(
            "Observable interaction patterns only. 2-4 short sentences describing "
            "what an external observer could see, in timeline order. No causal "
            "language, no interpretation, no system internals. 15-60 words."
        )
    )
    domain: str = pydantic.Field(
        description=(
            "The user's task area or goal. 1-10 word noun phrase, generalizable "
            "category. Not system area. e.g. 'music playback control'"
        )
    )
    severity: int = pydantic.Field(
        description="Severity: 1=minor, 3=moderate, 5=critical"
    )


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


class _ScorerSpec(pydantic.BaseModel):
    name: str = pydantic.Field(
        description="snake_case name for this scorer (should match the issue name if single scorer)"
    )
    detection_instructions: str = pydantic.Field(
        description=(
            "Instructions for a judge to detect this issue from a {{ trace }}. "
            "MUST contain the literal text '{{ trace }}'."
        )
    )


class _ScorerInstructionsResult(pydantic.BaseModel):
    scorers: list[_ScorerSpec] = pydantic.Field(
        description=(
            "One or more scorers to detect this issue. If the issue involves multiple "
            "independent criteria (e.g. 'X and Y' or 'X or Y'), split into separate scorers."
        )
    )


class _IdentifiedIssue(pydantic.BaseModel):
    name: str = pydantic.Field(
        description=(
            "Short readable title (3-8 words) followed by domain keywords in brackets, "
            "e.g. 'Media control commands ignored [music, spotify]'"
        )
    )
    description: str = pydantic.Field(description="What the issue is")
    root_cause: str = pydantic.Field(description="Why this issue occurs")
    example_indices: list[int] = pydantic.Field(
        description="Indices into the input trace summary list that exemplify this issue"
    )
    confidence: int = pydantic.Field(
        description=(
            "Confidence that this is a real, distinct issue (0-100). "
            "0=not confident, 25=might be real, 50=moderately confident, "
            "75=highly confident and real, 100=absolutely certain"
        ),
    )


