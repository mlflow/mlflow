"""
Cost anomalies, measured against an agent's own baseline.

Cost is the one category the triage judge cannot reach. Every other category is decided from
one trace's content, but "expensive" only means anything relative to the agent's other traces
-- a retrieval agent and a summarizer have different normal spend, so an experiment-wide
average hides both -- and a judge scoring one trace at a time has no view of the sample. The
arithmetic therefore lives here, and the finding it produces joins the others at clustering.

What counts as *how bad* is deliberately not decided here: a flagged trace enters triage as a
failing trace, and the cluster summariser assigns severity with the same rubric it applies to
every other finding.

Only a recorded ``total_cost`` is compared; missing usage is never read as zero spend. Where
a provider folds cache reads into prompt tokens the recorded cost overstates a cached request,
but every trace in a label is priced the same way, so the error largely cancels within the
comparison that is actually made.

This is the cost of the traced application. The cost of running detection itself is tracked
separately by ``_TokenCounter`` and never enters these baselines.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from mlflow.entities.trace import Trace
from mlflow.genai.discovery.utils import safe_trace_id
from mlflow.tracing.constant import TraceTagKey

_logger = logging.getLogger(__name__)

# How many same-label peers a baseline needs before it means anything. Below this, a "3 sigma"
# threshold is an artifact of the sample rather than a property of the agent.
MIN_BASELINE_PEERS = 5
# Standard deviations above the label's mean at which spend is called anomalous, per the
# category rubric's "roughly three standard deviations" guidance.
ANOMALY_SIGMA = 3.0

# Reason codes for traces a baseline could not price.
EXCLUDED_NO_COST = "no_cost_recorded"
EXCLUDED_FEW_PEERS = "too_few_same_label_peers"


@dataclass
class CostBaseline:
    """Normal spend for one agent label, computed from the sampled traces carrying it."""

    label: str
    peer_count: int
    mean_usd: float
    stdev_usd: float
    median_usd: float

    def sigma_above(self, cost_usd: float) -> float:
        """How many standard deviations above the mean a cost sits."""
        if self.stdev_usd <= 0:
            # Identical peers: any excess is unbounded in sigma terms, so fall back to a
            # ratio against the mean rather than dividing by zero.
            return math.inf if cost_usd > self.mean_usd else 0.0
        return (cost_usd - self.mean_usd) / self.stdev_usd


@dataclass
class CostAnomaly:
    """One trace whose spend is anomalous for its own agent label."""

    trace_id: str
    label: str
    cost_usd: float
    baseline: CostBaseline
    sigma: float

    def describe(self) -> str:
        """The anomaly as one clause, stating what the spend is compared against."""
        return (
            f"${self.cost_usd:.4f} against a typical ${self.baseline.median_usd:.4f} "
            f"for {self.label!r} ({self.baseline.peer_count} peers)"
        )


@dataclass
class CostAnalysis:
    """
    Cost findings for a sample, with everything the baselines could not price.

    Args:
        baselines: Per-label baselines that met the minimum peer count.
        anomalies: Traces at or above the anomaly threshold for their label.
        excluded: Trace ID to reason code for traces no baseline could price.
    """

    baselines: dict[str, CostBaseline] = field(default_factory=dict)
    anomalies: list[CostAnomaly] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)

    @property
    def anomalous_trace_ids(self) -> list[str]:
        return [anomaly.trace_id for anomaly in self.anomalies]

    def describe_limits(self) -> str:
        """What the cost analysis could not cover, for a coverage report."""
        if not self.excluded:
            return ""
        counts: dict[str, int] = {}
        for reason in self.excluded.values():
            counts[reason] = counts.get(reason, 0) + 1
        phrasing = {
            EXCLUDED_NO_COST: "with no recorded cost",
            EXCLUDED_FEW_PEERS: f"whose agent label had fewer than {MIN_BASELINE_PEERS} peers",
        }
        parts = [
            f"{count} {phrasing.get(reason, reason)}"
            for reason, count in sorted(counts.items())
            if count
        ]
        return f"Cost analysis skipped {', '.join(parts)}."


def get_agent_label(trace: Trace) -> str:
    """
    The label whose peers a trace's spend is compared against.

    OSS traces carry no dedicated agent field, so the trace name is used: it is the root span's
    name, which is the entry point the application was called through, and is what distinguishes
    a retrieval agent from a summarizer in practice. Traces with no name share one bucket rather
    than being compared against unrelated work.
    """
    tags = trace.info.tags or {}
    if name := tags.get(TraceTagKey.TRACE_NAME):
        # Coerced because the label only has to group and be printable: a tag written with a
        # non-string value should bucket the trace, not raise while baselining the sample.
        return str(name)
    # Fall back to the root span's name, which is what the tag is derived from.
    for span in trace.data.spans or []:
        if span.parent_id is None and span.name:
            return str(span.name)
    return "(unnamed)"


def _trace_cost(trace: Trace) -> float | None:
    cost = trace.info.cost or {}
    total = cost.get("total_cost")
    return float(total) if total else None


def _stdev(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _baseline_from(label: str, costs: list[float]) -> CostBaseline:
    mean = sum(costs) / len(costs) if costs else 0.0
    return CostBaseline(
        label=label,
        peer_count=len(costs),
        mean_usd=mean,
        stdev_usd=_stdev(costs, mean),
        median_usd=_median(costs) if costs else 0.0,
    )


def analyze_cost_anomalies(
    traces: list[Trace],
    *,
    min_peers: int = MIN_BASELINE_PEERS,
) -> CostAnalysis:
    """
    Flag traces whose spend is anomalous against their own agent label's baseline.

    Args:
        traces: The sampled traces. Only recorded cost and usage are read.
        min_peers: How many same-label traces a baseline needs. Labels below this are
            reported as excluded rather than compared against a meaningless baseline.

    Returns:
        A :class:`CostAnalysis` with per-label baselines, the anomalies, and the traces no
        baseline could price. Missing usage is never treated as zero spend.
    """
    analysis = CostAnalysis()

    by_label: dict[str, list[tuple[str, float]]] = {}
    for position, trace in enumerate(traces):
        trace_id = safe_trace_id(trace, position)
        try:
            cost = _trace_cost(trace)
            label = get_agent_label(trace)
        except Exception:
            # Cost findings are advisory, so a trace whose usage cannot be read is disclosed
            # as unpriced rather than taking the run down with it.
            _logger.debug("Could not read cost for trace %s", trace_id, exc_info=True)
            analysis.excluded[trace_id] = EXCLUDED_NO_COST
            continue
        if cost is None:
            analysis.excluded[trace_id] = EXCLUDED_NO_COST
            continue
        by_label.setdefault(label, []).append((trace_id, cost))

    for label, entries in sorted(by_label.items(), key=lambda item: str(item[0])):
        if len(entries) < min_peers:
            for trace_id, _ in entries:
                analysis.excluded[trace_id] = EXCLUDED_FEW_PEERS
            continue

        costs = [cost for _, cost in entries]
        analysis.baselines[label] = _baseline_from(label, costs)

        for trace_id, cost in entries:
            # Baseline the candidate against the *other* traces carrying its label. Leaving it
            # in would let one expensive trace raise the mean and the deviation it is then
            # measured against -- with a sample this size a lone outlier could not reach the
            # threshold at all, so the anomaly would hide itself.
            peer_costs = [peer_cost for peer_id, peer_cost in entries if peer_id != trace_id]
            peer_baseline = _baseline_from(label, peer_costs)
            sigma = peer_baseline.sigma_above(cost)
            if sigma < ANOMALY_SIGMA:
                continue
            analysis.anomalies.append(
                CostAnomaly(
                    trace_id=trace_id,
                    label=label,
                    cost_usd=cost,
                    baseline=peer_baseline,
                    sigma=sigma,
                )
            )

    return analysis
