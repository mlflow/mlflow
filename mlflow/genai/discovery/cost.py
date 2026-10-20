"""
Cost anomalies, measured against an agent's own baseline.

Cost is the one category where the expensive traces are usually not the issue: a retrieval
agent and a summarizer have different normal spend, so an experiment-wide average hides both.
This module baselines spend per agent label and flags only traces that are anomalous for
*their own* label, then attributes the extra spend to what the flagged traces do differently.

This is the cost of the traced application. The cost of running detection itself is tracked
separately by ``_TokenCounter`` and never enters these baselines.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from mlflow.entities.span import SpanType
from mlflow.entities.trace import Trace
from mlflow.tracing.constant import TraceTagKey

_logger = logging.getLogger(__name__)

# How many same-label peers a baseline needs before it means anything. Below this, a "3 sigma"
# threshold is an artifact of the sample rather than a property of the agent.
MIN_BASELINE_PEERS = 5
# Standard deviations above the label's mean at which spend is called anomalous, per the
# category rubric's "roughly three standard deviations" guidance.
ANOMALY_SIGMA = 3.0
# Spend above the label's baseline but below the anomaly threshold is reported as low severity
# rather than ignored, starting at this many standard deviations.
ELEVATED_SIGMA = 1.5
# A driver this many times the label's typical call count is treated as the extra work.
DRIVER_RATIO = 2.0
# Repeating one identical call this many times reads as a retry or loop rather than fixed
# extra work, which is what separates an unbounded driver from a bounded one.
REPEAT_LOOP_THRESHOLD = 3

# Reason codes for traces a baseline could not price.
EXCLUDED_NO_COST = "no_cost_recorded"
EXCLUDED_FEW_PEERS = "too_few_same_label_peers"


@dataclass
class CostDriver:
    """
    What a flagged trace does that its baseline does not.

    Args:
        description: The driver as a human-readable comparison, e.g. "22 tool calls against a
            typical 5".
        is_bounded: Whether the extra work is a fixed amount per trace. An unbounded driver --
            an uncapped retry or tool-call loop -- has a worst case the sampled traces do not
            bound, which the severity rubric treats as high.
    """

    description: str
    is_bounded: bool


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
    drivers: list[CostDriver] = field(default_factory=list)

    @property
    def is_anomalous(self) -> bool:
        return self.sigma >= ANOMALY_SIGMA

    @property
    def has_unbounded_driver(self) -> bool:
        return any(not driver.is_bounded for driver in self.drivers)

    @property
    def severity(self) -> str:
        """
        Severity per the category rubric.

        High when the driver is unbounded (an uncapped retry or loop, whose worst case the
        sample does not bound), medium when it is a fixed amount of extra work, and low when
        spend is above the label's baseline but below the anomaly threshold.
        """
        if not self.is_anomalous:
            return "low"
        return "high" if self.has_unbounded_driver else "medium"

    def describe(self) -> str:
        """The anomaly as one sentence, stating what the spend is compared against."""
        comparison = (
            f"${self.cost_usd:.4f} against a typical ${self.baseline.median_usd:.4f} "
            f"for {self.label!r}"
        )
        if not self.drivers:
            return comparison
        return f"{comparison}; {'; '.join(driver.description for driver in self.drivers)}"


@dataclass
class CostAnalysis:
    """
    Cost findings for a sample, with everything the baselines could not price.

    Args:
        baselines: Per-label baselines that met the minimum peer count.
        anomalies: Traces at or above the anomaly threshold for their label.
        elevated: Traces above their label's baseline but below the threshold.
        excluded: Trace ID to reason code for traces no baseline could price.
    """

    baselines: dict[str, CostBaseline] = field(default_factory=dict)
    anomalies: list[CostAnomaly] = field(default_factory=list)
    elevated: list[CostAnomaly] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)

    @property
    def anomalous_trace_ids(self) -> list[str]:
        return [anomaly.trace_id for anomaly in self.anomalies]

    def describe_limits(self) -> str:
        """What the cost analysis could not cover, for a coverage report."""
        if not self.excluded:
            return ""
        no_cost = sum(1 for reason in self.excluded.values() if reason == EXCLUDED_NO_COST)
        few_peers = sum(1 for reason in self.excluded.values() if reason == EXCLUDED_FEW_PEERS)
        parts = []
        if no_cost:
            parts.append(f"{no_cost} with no recorded cost")
        if few_peers:
            parts.append(f"{few_peers} whose agent label had fewer than {MIN_BASELINE_PEERS} peers")
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


def _call_counts(trace: Trace) -> dict[str, int]:
    """Calls per span type, which is the unit a cost driver is quoted in."""
    counts: dict[str, int] = {}
    for span in trace.data.spans or []:
        counts[span.span_type] = counts.get(span.span_type, 0) + 1
    return counts


def _repeated_call_names(trace: Trace) -> dict[str, int]:
    """How many times each span name repeats, which is how a retry or loop shows up."""
    names: dict[str, int] = {}
    for span in trace.data.spans or []:
        if span.span_type in (SpanType.TOOL, SpanType.RETRIEVER, SpanType.LLM, SpanType.CHAT_MODEL):
            names[span.name] = names.get(span.name, 0) + 1
    return {name: count for name, count in names.items() if count > 1}


def _attribute_drivers(trace: Trace, peers: list[Trace]) -> list[CostDriver]:
    """
    What the flagged trace does more of than its peers.

    A driver is only reported when it can be quantified against the baseline, because a cost
    finding that does not say what the trace is expensive *compared to* is not actionable.
    """
    drivers: list[CostDriver] = []
    trace_counts = _call_counts(trace)

    # Typical call count per span type across the peers, so the comparison is against the
    # label's own normal rather than an absolute number.
    peer_counts: dict[str, list[int]] = {}
    for peer in peers:
        for span_type, count in _call_counts(peer).items():
            peer_counts.setdefault(span_type, []).append(count)

    for span_type, count in sorted(trace_counts.items()):
        typical_values = peer_counts.get(span_type, [])
        typical = _median(typical_values) if typical_values else 0.0
        if typical <= 0:
            continue
        if count >= typical * DRIVER_RATIO and count - typical >= 1:
            label = span_type.lower().replace("_", " ")
            drivers.append(
                CostDriver(
                    description=(f"{count} {label} calls against a typical {typical:.0f}"),
                    # Extra calls spread across distinct operations are a fixed amount of
                    # additional work; a repeated identical call is a retry or loop, which the
                    # sample does not bound.
                    is_bounded=True,
                )
            )

    # A repeated identical call overrides the bounded reading: its worst case is whatever the
    # loop does next, not what this sample happened to spend.
    for name, repeats in sorted(_repeated_call_names(trace).items()):
        if repeats >= REPEAT_LOOP_THRESHOLD:
            drivers.append(
                CostDriver(
                    description=f"repeats {name!r} {repeats} times",
                    is_bounded=False,
                )
            )

    return drivers


def analyze_cost_anomalies(
    traces: list[Trace],
    *,
    min_peers: int = MIN_BASELINE_PEERS,
) -> CostAnalysis:
    """
    Flag traces whose spend is anomalous against their own agent label's baseline.

    Args:
        traces: The sampled traces. Only recorded cost and span structure are read.
        min_peers: How many same-label traces a baseline needs. Labels below this are
            reported as excluded rather than compared against a meaningless baseline.

    Returns:
        A :class:`CostAnalysis` with per-label baselines, the anomalies and their drivers, and
        the traces no baseline could price. Missing usage is never treated as zero spend.
    """
    analysis = CostAnalysis()

    by_label: dict[str, list[tuple[Trace, float]]] = {}
    for trace in traces:
        cost = _trace_cost(trace)
        if cost is None:
            analysis.excluded[trace.info.trace_id] = EXCLUDED_NO_COST
            continue
        by_label.setdefault(get_agent_label(trace), []).append((trace, cost))

    for label, entries in sorted(by_label.items(), key=lambda item: str(item[0])):
        if len(entries) < min_peers:
            for trace, _ in entries:
                analysis.excluded[trace.info.trace_id] = EXCLUDED_FEW_PEERS
            continue

        costs = [cost for _, cost in entries]
        analysis.baselines[label] = _baseline_from(label, costs)

        for trace, cost in entries:
            # Baseline the candidate against the *other* traces carrying its label. Leaving it
            # in would let one expensive trace raise the mean and the deviation it is then
            # measured against -- with a sample this size a lone outlier could not reach the
            # threshold at all, so the anomaly would hide itself.
            peer_entries = [
                (peer, peer_cost)
                for peer, peer_cost in entries
                if peer.info.trace_id != trace.info.trace_id
            ]
            peer_baseline = _baseline_from(label, [peer_cost for _, peer_cost in peer_entries])
            sigma = peer_baseline.sigma_above(cost)
            if sigma < ELEVATED_SIGMA:
                continue
            anomaly = CostAnomaly(
                trace_id=trace.info.trace_id,
                label=label,
                cost_usd=cost,
                baseline=peer_baseline,
                sigma=sigma,
                drivers=(
                    _attribute_drivers(trace, [peer for peer, _ in peer_entries])
                    if sigma >= ANOMALY_SIGMA
                    else []
                ),
            )
            if anomaly.is_anomalous:
                analysis.anomalies.append(anomaly)
            else:
                analysis.elevated.append(anomaly)

    return analysis
