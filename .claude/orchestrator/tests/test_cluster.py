from __future__ import annotations

import json

import pytest
from orchestrator.cluster import (
    Cluster,
    SourceFinding,
    clusters_to_drafts,
    materialize_clusters,
    parse_cluster_assignments,
)


def _src(
    source: str = "reviewer_standalone",
    path: str = "f.py",
    line: int = 100,
    body: str = "concern X",
    severity: str = "ask",
    rule_id: str | None = "GEN-4",
) -> SourceFinding:
    return SourceFinding(
        source=source,
        path=path,
        line=line,
        body=body,
        severity=severity,
        rule_id=rule_id,
    )


def test_parse_cluster_assignments_basic() -> None:
    judge_output = json.dumps({"clusters": [{"a": [0], "b": [0]}, {"a": [1], "b": []}]})
    result = parse_cluster_assignments(judge_output, n_a=2, n_b=1)
    assert result == [[("a", 0), ("b", 0)], [("a", 1)]]


def test_parse_cluster_assignments_invalid_json() -> None:
    with pytest.raises(ValueError, match="non-JSON"):
        parse_cluster_assignments("not json", n_a=1, n_b=1)


def test_parse_cluster_assignments_duplicate_index_rejected() -> None:
    judge_output = json.dumps({"clusters": [{"a": [0]}, {"a": [0]}]})
    with pytest.raises(ValueError, match="duplicate"):
        parse_cluster_assignments(judge_output, n_a=1, n_b=0)


def test_parse_cluster_assignments_out_of_range_rejected() -> None:
    judge_output = json.dumps({"clusters": [{"a": [5]}]})
    with pytest.raises(ValueError, match="duplicate|Invalid"):
        parse_cluster_assignments(judge_output, n_a=1, n_b=0)


def test_parse_cluster_assignments_missing_indices_rejected() -> None:
    judge_output = json.dumps({"clusters": [{"a": [0]}]})
    with pytest.raises(ValueError, match="missed indices"):
        parse_cluster_assignments(judge_output, n_a=2, n_b=0)


def test_materialize_clusters_zips_indices_to_findings() -> None:
    a = [_src(body="A0"), _src(body="A1")]
    b = [_src(source="reviewer_opinion", body="B0")]
    judge_output = json.dumps({"clusters": [{"a": [0], "b": [0]}, {"a": [1], "b": []}]})
    clusters = materialize_clusters(judge_output, a, b)
    assert len(clusters) == 2
    bodies_first = sorted(m.body for m in clusters[0].members)
    assert bodies_first == ["A0", "B0"]
    assert [m.body for m in clusters[1].members] == ["A1"]


def test_cluster_representative_body_prefers_opinion() -> None:
    standalone = _src(source="reviewer_standalone", body="standalone-body")
    opinion = _src(source="reviewer_opinion", body="opinion-body")
    cluster = Cluster(members=[standalone, opinion])
    assert cluster.representative_body == "opinion-body"


def test_cluster_representative_body_falls_back_to_standalone() -> None:
    standalone = _src(source="reviewer_standalone", body="standalone-body")
    cluster = Cluster(members=[standalone])
    assert cluster.representative_body == "standalone-body"


@pytest.mark.parametrize(
    ("severities", "expected"),
    [
        (["nit"], "nit"),
        (["nit", "ask"], "ask"),
        (["nit", "ask", "firm"], "firm"),
        (["nit", "ask", "firm", "block"], "block"),
        (["block", "firm"], "block"),
        (["unknown", "ask"], "ask"),
    ],
)
def test_cluster_representative_severity(severities: list[str], expected: str) -> None:
    members = [_src(severity=s) for s in severities]
    cluster = Cluster(members=members)
    assert cluster.representative_severity == expected


def test_clusters_to_drafts_uses_representatives() -> None:
    cluster = Cluster(
        members=[
            _src(source="reviewer_standalone", body="A", line=10, severity="nit"),
            _src(source="reviewer_opinion", body="B", line=10, severity="firm"),
        ]
    )
    drafts = clusters_to_drafts([cluster])
    assert len(drafts) == 1
    assert drafts[0].body == "B"
    assert drafts[0].severity == "firm"
    assert drafts[0].source == "cluster"


def test_clusters_to_drafts_empty_input() -> None:
    assert clusters_to_drafts([]) == []
