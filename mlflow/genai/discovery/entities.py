from __future__ import annotations

import threading
from dataclasses import dataclass

import pydantic

from mlflow.genai.discovery.constants import ConfidenceLevel


class _TokenCounter:
    """Thread-safe accumulator for LLM token usage across pipeline phases."""

    def __init__(self):
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0

    def track(self, response) -> None:
        with self._lock:
            usage = getattr(response, "usage", None)
            if usage:
                self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
            if hidden := getattr(response, "_hidden_params", None):
                if cost := hidden.get("response_cost"):
                    self.cost_usd += cost

    def to_dict(self) -> dict:
        result = {}
        total = self.input_tokens + self.output_tokens
        if total > 0:
            result["input_tokens"] = self.input_tokens
            result["output_tokens"] = self.output_tokens
            result["total_tokens"] = total
        if self.cost_usd > 0:
            result["cost_usd"] = round(self.cost_usd, 6)
        return result


@dataclass
class Issue:
    issue_id: str
    run_id: str
    name: str
    description: str
    root_cause: str
    example_trace_ids: list[str]
    frequency: float
    confidence: ConfidenceLevel
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
    confidence: ConfidenceLevel = pydantic.Field(
        description=(
            "Confidence that this is a real, distinct issue. "
            "definitely_no=not a real issue, weak_no=probably not real, "
            "maybe=uncertain, weak_yes=probably real, definitely_yes=certainly real"
        ),
    )
