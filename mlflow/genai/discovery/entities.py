from __future__ import annotations

import math
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
            the trace, so these are excluded from trace-quality classification rather than
            counted as passing.
        unclassified_trace_ids: Traces a scorer returned no value for, so neither pass nor
            fail was established.
        scorer_errors: Trace ID to the scorer's error message, for the traces in
            ``scorer_error_trace_ids``.

    The last three are surfaced in the run's log and nowhere else today. They are the input a
    coverage report would need -- how much of the sample the run is actually entitled to draw
    conclusions from -- but that surface is deferred, so nothing reads them programmatically.
    """

    failing_traces: list[Trace]
    rationale_map: dict[str, str]
    categories_map: dict[str, list[str]]
    scorer_error_trace_ids: list[str] = field(default_factory=list)
    unclassified_trace_ids: list[str] = field(default_factory=list)
    scorer_errors: dict[str, str] = field(default_factory=dict)


def _format_usd(value: float) -> str:
    """
    A dollar figure that keeps two significant digits however small it is.

    Four decimals is right for ordinary spend but silently flattens cheap traces: a $0.0000014
    anomaly against a $0.000001 baseline renders as "$0.0000 ... the typical $0.0000", which
    reaches the clustering prompt as two identical figures and reads as no anomaly at all.
    """
    if not math.isfinite(value) or value <= 0:
        return f"${value:.4f}"
    return f"${value:.{max(4, -math.floor(math.log10(value)) + 1)}f}"


@dataclass
class CostAnomaly:
    """
    One trace whose spend is anomalous for the sample it was drawn from.

    Args:
        trace_id: The trace this describes.
        label: The agent label whose peers it was compared against.
        cost_usd: What the trace cost.
        typical_usd: The median spend of its comparison group.
        peer_count: How many priced traces the comparison group held.
        deviations: How far above typical the spend sits, in robust deviations of
            log spend. Reported so a reader can tell a clear outlier from a marginal one.
            Infinite when the peers agreed on one figure, leaving no spread to measure.
    """

    trace_id: str
    label: str
    cost_usd: float
    typical_usd: float
    peer_count: int
    deviations: float

    def describe(self) -> str:
        """The anomaly as one clause, stating what the spend is compared against."""
        multiple = self.cost_usd / self.typical_usd if self.typical_usd > 0 else float("inf")
        # A fixed multiple is rounded too, so a 1.4x anomaly does not read as "1x the typical".
        multiple_text = f"{multiple:.0f}" if multiple >= 10 else f"{multiple:.1f}"
        return (
            f"{_format_usd(self.cost_usd)}, {multiple_text}x the typical "
            f"{_format_usd(self.typical_usd)} for {self.label!r} across "
            f"{self.peer_count} priced traces"
        )


@dataclass
class DiscoverIssuesResult:
    issues: list[Issue]
    triage_run_id: str
    summary: str
    total_traces_analyzed: int
    total_cost_usd: float | None = None
    # How many of `issues` this run filed; the rest were already reported and were refreshed
    # rather than duplicated.
    new_issue_count: int = 0


_SIGNAL_SEPARATOR = "; "


def _share_budget(text: str, limit: int) -> str:
    """
    ``text`` trimmed to ``limit`` characters, keeping the head of every ``; ``-joined signal.

    Signals are trimmed from the longest down, so short ones survive whole and the budget is
    spent on them only as far as they need it. Returned joined the way it arrived, so callers
    and prompts see the same shape as before.
    """
    if len(text) <= limit:
        return text
    signals = text.split(_SIGNAL_SEPARATOR)
    if len(signals) == 1:
        return text[:limit]

    # Every signal needs room for itself plus the separators between them.
    budget = limit - len(_SIGNAL_SEPARATOR) * (len(signals) - 1)
    if budget <= 0:
        return text[:limit]

    kept = [0] * len(signals)
    remaining = budget
    # Hand out the budget in equal rounds: a signal shorter than its share takes only what it
    # needs and leaves the rest, so one long signal cannot starve the others.
    for index in sorted(range(len(signals)), key=lambda i: len(signals[i])):
        share = remaining // max(1, len(signals) - sum(1 for k in kept if k))
        take = min(len(signals[index]), max(share, 0))
        kept[index] = take
        remaining -= take
    return _SIGNAL_SEPARATOR.join(signal[:take] for signal, take in zip(signals, kept) if take > 0)


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
        """
        The rationale trimmed to what the labeling and summarization prompts will carry.

        Each ``; ``-joined signal keeps a share of the budget rather than the whole string
        being cut at one point. A blind prefix cut silently dropped whichever signal came
        last: a long session feedback hid the turn's own triage rationale, and appended cost
        facts ("$0.40 against a $0.01 baseline") vanished behind a long quality rationale,
        leaving the clustering input with the ``cost`` category and no figures to cluster on.
        """
        return _share_budget(self.full_rationale, RATIONALE_TRUNCATION_LIMIT)


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
