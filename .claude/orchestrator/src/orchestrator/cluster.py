"""Merge findings from multiple sources into deduplicated clusters.

In default mode the orchestrator collects findings from three sources:

- `reviewer_standalone`: the mlflow-reviewer agent run in discovery mode
- `reviewer_opinion`: the mlflow-reviewer agent's opinions on spotter findings
  that it chose to raise
- `spotter` (only via `reviewer_opinion`): the spotter findings the reviewer
  raised; carried through with their `finding_id` for traceability

Two findings cluster together if they describe the SAME concern (same root
cause / same suggested fix). The cluster judge is an LLM call that does the
matching; this module wraps the call and produces canonical cluster output.

The matching approach mirrors the eval pipeline at
`databricks-misc/per-reviewer-rulesets/scripts/three_way/`: feed the lists to
a judge, get back cluster assignments by index, materialize one DraftFinding
per cluster (preferring the reviewer_standalone or reviewer_opinion body
since those are already in team voice).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from orchestrator.dedup import DraftFinding

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceFinding:
    source: str  # "reviewer_standalone" | "reviewer_opinion"
    path: str
    line: int
    body: str
    severity: str
    rule_id: str | None


@dataclass
class Cluster:
    members: list[SourceFinding] = field(default_factory=list)

    @property
    def representative_body(self) -> str:
        """Pick the body that goes into the posted comment.

        Prefer reviewer_opinion when it exists (it's the per-finding rewrite the
        reviewer chose for the spotter discovery). Fall back to
        reviewer_standalone.
        """
        opinion = next((m for m in self.members if m.source == "reviewer_opinion"), None)
        if opinion is not None:
            return opinion.body
        return self.members[0].body

    @property
    def representative_severity(self) -> str:
        priority = {"block": 4, "firm": 3, "ask": 2, "nit": 1}
        return max(self.members, key=lambda m: priority.get(m.severity, 0)).severity

    @property
    def representative_path(self) -> str:
        return self.members[0].path

    @property
    def representative_line(self) -> int:
        return self.members[0].line

    @property
    def rule_id(self) -> str | None:
        return next((m.rule_id for m in self.members if m.rule_id), None)


def parse_cluster_assignments(judge_output: str, n_a: int, n_b: int) -> list[Cluster]:
    """Parse the cluster judge's JSON output into Cluster objects.

    Judge output schema:
      {"clusters": [{"a": [<idx>...], "b": [<idx>...]}, ...]}

    Each entry is a cluster; each list holds 0+ indices into the corresponding
    input list. Every input index must appear in exactly one cluster.
    """
    try:
        data = json.loads(judge_output)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cluster judge returned non-JSON output: {e}") from e

    raw_clusters = data.get("clusters", [])
    if not isinstance(raw_clusters, list):
        raise ValueError(f"`clusters` must be a list, got {type(raw_clusters).__name__}")

    seen_a: set[int] = set()
    seen_b: set[int] = set()
    out: list[list[tuple[str, int]]] = []
    for entry in raw_clusters:
        a_idx = entry.get("a", []) or []
        b_idx = entry.get("b", []) or []
        members: list[tuple[str, int]] = []
        for i in a_idx:
            if i in seen_a or not (0 <= i < n_a):
                raise ValueError(f"Invalid or duplicate `a` index {i}")
            seen_a.add(i)
            members.append(("a", i))
        for i in b_idx:
            if i in seen_b or not (0 <= i < n_b):
                raise ValueError(f"Invalid or duplicate `b` index {i}")
            seen_b.add(i)
            members.append(("b", i))
        out.append(members)

    missing_a = set(range(n_a)) - seen_a
    missing_b = set(range(n_b)) - seen_b
    if missing_a or missing_b:
        raise ValueError(
            f"Cluster judge missed indices: a={sorted(missing_a)}, b={sorted(missing_b)}"
        )
    return out  # caller materializes Cluster objects from the source lists


def materialize_clusters(
    judge_output: str,
    list_a: list[SourceFinding],
    list_b: list[SourceFinding],
) -> list[Cluster]:
    raw = parse_cluster_assignments(judge_output, len(list_a), len(list_b))
    clusters: list[Cluster] = []
    for entry in raw:
        cluster = Cluster()
        for source, idx in entry:
            cluster.members.append(list_a[idx] if source == "a" else list_b[idx])
        clusters.append(cluster)
    return clusters


def clusters_to_drafts(clusters: list[Cluster]) -> list[DraftFinding]:
    return [
        DraftFinding(
            path=c.representative_path,
            line=c.representative_line,
            body=c.representative_body,
            severity=c.representative_severity,
            rule_id=c.rule_id,
            source="cluster",
        )
        for c in clusters
    ]
