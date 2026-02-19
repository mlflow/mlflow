from __future__ import annotations

import pydantic


class _TraceAnalysis(pydantic.BaseModel):
    trace_index: int = pydantic.Field(description="Index of the trace in the input list")
    failure_category: str = pydantic.Field(
        description=(
            "Category of failure: tool_error, hallucination, latency, "
            "incomplete_response, error_propagation, wrong_tool_use, "
            "context_loss, or other"
        )
    )
    failure_summary: str = pydantic.Field(description="Brief summary of what went wrong")
    root_cause_hypothesis: str = pydantic.Field(
        description="Hypothesis about why this failure occurred based on span-level evidence"
    )
    affected_spans: list[str] = pydantic.Field(
        description="Names of spans most relevant to the failure"
    )
    severity: int = pydantic.Field(
        description="Severity of the failure (1=minor, 3=moderate, 5=critical)"
    )


class _BatchTraceAnalysisResult(pydantic.BaseModel):
    analyses: list[_TraceAnalysis] = pydantic.Field(description="Analysis for each failing trace")


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
    name: str = pydantic.Field(description="snake_case identifier for the issue")
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


class _IssueClusteringResult(pydantic.BaseModel):
    issues: list[_IdentifiedIssue] = pydantic.Field(
        description="Distinct issues identified from the failing traces"
    )
