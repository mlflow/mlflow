from __future__ import annotations

from dataclasses import dataclass, field

from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.scorers.base import Scorer


@dataclass
class Issue:
    """A distinct issue discovered in the experiment's traces."""

    name: str
    description: str
    root_cause: str
    example_trace_ids: list[str]
    scorer: Scorer
    frequency: float
    confidence: int
    rationale_examples: list[str] = field(default_factory=list)


@dataclass
class DiscoverIssuesResult:
    """Result of the discover_issues pipeline."""

    issues: list[Issue]
    triage_evaluation: EvaluationResult
    validation_evaluation: EvaluationResult | None
    summary: str
    total_traces_analyzed: int
