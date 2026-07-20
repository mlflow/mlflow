# ruff: noqa: T201
"""Cluster likely-duplicate security advisories for maintainer review.

Notify-only: this command performs read-only GET requests and never mutates any
advisory. It surfaces candidate duplicate clusters and highlights the earliest
report in each; the maintainer decides what to do.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from skills.github import GitHubClient, SecurityAdvisory

STATES = ["triage", "draft", "published", "closed"]

# Structural boosts only nudge pairs whose text is already fairly similar (cosine
# >= COSINE_FLOOR) over the threshold; they never merge text-dissimilar reports.
# This keeps text similarity the dominant signal and prevents a generic shared
# tag (e.g. "artifact") from chaining unrelated reports into one giant cluster.
COSINE_FLOOR = 0.45
CWE_BOOST = 0.1
COMPONENT_BOOST = 0.1

# Curated mlflow-relevant component / vulnerability-class keywords. Each advisory
# is tagged with the component names whose keywords appear in its text; sharing a
# tag is a strong duplicate signal.
COMPONENTS: dict[str, list[str]] = {
    "mlflow server/ui": ["mlflow server", "mlflow ui", "mlflow.server"],
    "gateway": ["gateway", "deployments server"],
    "model registry": ["model registry"],
    "artifact": ["artifact"],
    "graphql": ["graphql"],
    "rest api": ["/ajax-api", "/api/2.0", "rest api"],
    "path traversal": [
        "path traversal",
        "directory traversal",
        "local file inclusion",
        "lfi",
        "arbitrary file",
        "file access",
    ],
    "ssrf": ["ssrf", "server-side request forgery", "server side request forgery"],
    "rce": ["remote code execution", "arbitrary code execution", " rce"],
    "deserialization": ["deserialization", "pickle", "yaml.load", "unsafe yaml"],
    "access control": [
        "authentication bypass",
        "authorization",
        "access control",
        "bola",
        "idor",
        "broken access",
        "privilege escalation",
    ],
    "xss": ["cross-site scripting", "xss"],
    "sql injection": ["sql injection", "sqli"],
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "be",
    "by",
    "as",
    "at",
    "this",
    "that",
    "it",
    "its",
    "from",
    "can",
    "could",
    "would",
    "when",
    "which",
    "via",
    "using",
    "use",
    "used",
    "allows",
    "allow",
    "vulnerability",
    "vulnerable",
    "issue",
    "security",
    "attacker",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def parse_repo(repo: str) -> tuple[str, str]:
    owner, _, name = repo.partition("/")
    if not owner or not name:
        log(f"Error: Invalid repo '{repo}'. Expected 'owner/repo'.")
        sys.exit(1)
    return owner, name


def advisory_text(advisory: SecurityAdvisory) -> str:
    return f"{advisory.summary}\n{advisory.description or ''}".lower()


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text) if t not in STOPWORDS and len(t) > 1]


def detect_components(text: str) -> frozenset[str]:
    return frozenset(
        name for name, keywords in COMPONENTS.items() if any(k in text for k in keywords)
    )


def build_tfidf(docs_tokens: list[list[str]]) -> list[dict[str, float]]:
    n = len(docs_tokens)
    df: Counter[str] = Counter()
    for tokens in docs_tokens:
        df.update(set(tokens))
    idf = {term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()}

    vectors: list[dict[str, float]] = []
    for tokens in docs_tokens:
        if not tokens:
            vectors.append({})
            continue
        tf = Counter(tokens)
        length = len(tokens)
        vectors.append({term: (freq / length) * idf[term] for term, freq in tf.items()})
    return vectors


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # Iterate over the smaller vector for the dot product; look terms up in the other.
    smaller = a if len(a) <= len(b) else b
    larger = b if smaller is a else a
    dot = sum(weight * larger.get(term, 0.0) for term, weight in smaller.items())
    if dot == 0.0:
        return 0.0
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def pair_score(
    i: int,
    j: int,
    vectors: list[dict[str, float]],
    components: list[frozenset[str]],
    cwes: list[frozenset[str]],
    advisories: list[SecurityAdvisory],
) -> float:
    # A shared CVE id is a near-definitive duplicate signal.
    cve_i = advisories[i].cve_id
    cve_j = advisories[j].cve_id
    if cve_i and cve_j and cve_i == cve_j:
        return 1.0

    score = cosine(vectors[i], vectors[j])
    # Only let structural signals adjust the score for pairs that are already
    # textually similar; otherwise a shared generic tag would merge unrelated
    # reports and chain them into oversized clusters.
    if score >= COSINE_FLOOR:
        if cwes[i] & cwes[j]:
            score += CWE_BOOST
        if components[i] & components[j]:
            score += COMPONENT_BOOST
    return score


def sort_key(advisory: SecurityAdvisory) -> tuple[str, str]:
    return (advisory.created_at or "", advisory.ghsa_id)


def cluster_advisories(
    advisories: list[SecurityAdvisory], threshold: float, min_cluster_size: int
) -> list[list[SecurityAdvisory]]:
    """Leader-clustering keyed on the earliest report.

    Reports are processed oldest-first; each is attached to the existing cluster
    whose representative (the earliest report, i.e. the "original") it is most
    similar to above ``threshold``, otherwise it becomes a new representative.
    Because membership is defined by similarity to the representative rather than
    to any neighbour, unrelated reports cannot chain into one oversized cluster.
    """
    docs_tokens = [tokenize(advisory_text(a)) for a in advisories]
    vectors = build_tfidf(docs_tokens)
    components = [detect_components(advisory_text(a)) for a in advisories]
    cwes = [frozenset(a.cwe_ids) for a in advisories]

    order = sorted(range(len(advisories)), key=lambda i: sort_key(advisories[i]))
    reps: list[int] = []
    members: dict[int, list[int]] = {}

    for i in order:
        best_rep: int | None = None
        best_score = threshold
        for rep in reps:
            score = pair_score(i, rep, vectors, components, cwes, advisories)
            if score >= best_score:
                best_score = score
                best_rep = rep
        if best_rep is None:
            reps.append(i)
            members[i] = [i]
        else:
            members[best_rep].append(i)

    clusters = [
        [advisories[idx] for idx in sorted(idxs, key=lambda k: sort_key(advisories[k]))]
        for idxs in members.values()
        if len(idxs) >= min_cluster_size
    ]
    # Largest clusters first; break ties by earliest report then GHSA id for determinism.
    clusters.sort(key=lambda cluster: (-len(cluster), sort_key(cluster[0])))
    return clusters


def format_member(advisory: SecurityAdvisory) -> str:
    date = (advisory.created_at or "")[:10] or "unknown date"
    reporter = advisory.author.login if advisory.author and advisory.author.login else "unknown"
    severity = advisory.severity or "unknown"
    cwes = ", ".join(advisory.cwe_ids) if advisory.cwe_ids else "no CWE"
    return (
        f"- {advisory.ghsa_id} | {date} | reporter: {reporter} | {severity} | {cwes}\n"
        f"  {advisory.html_url}\n"
        f"  {advisory.summary}"
    )


def format_singletons(singletons: list[SecurityAdvisory]) -> list[str]:
    lines = ["", f"## Not part of any duplicate cluster ({len(singletons)})", ""]
    if not singletons:
        lines.append("None - every analyzed advisory fell into a candidate cluster.")
        return lines
    lines.append(
        "No duplicate candidates detected for these; each needs to be reviewed independently."
    )
    lines.extend(format_member(a) for a in singletons)
    return lines


def format_member_markdown(advisory: SecurityAdvisory, *, original: bool = False) -> str:
    """Render one advisory as a Markdown list item: linked GHSA id, date, CWE(s), title.

    Every character here is derived mechanically from advisory fields, so the CLI
    can emit the full report scaffold without the model re-typing any link.
    """
    date = (advisory.created_at or "")[:10] or "unknown date"
    if advisory.cwe_ids:
        cwes = "CWE-" + "/".join(c.removeprefix("CWE-") for c in advisory.cwe_ids)
    else:
        cwes = "no CWE"
    title = (advisory.summary or "").strip().replace("\n", " ") or "(no title)"
    suffix = "  _(original)_" if original else ""
    return f"- [{advisory.ghsa_id}]({advisory.html_url}) ({date}, {cwes}) — {title}{suffix}"


def format_scaffold(
    clusters: list[list[SecurityAdvisory]],
    total: int,
    threshold: float,
    state: str,
    singletons: list[SecurityAdvisory],
) -> str:
    """Fully-rendered Markdown report scaffold for the model to enrich in place.

    Everything mechanical (title, counts, notify-only header, per-member links,
    the entire singleton section, bold original/duplicate labels) is emitted here.
    The model only fills the ``TODO`` markers (semantic titles, confidence,
    reasoning, tier index) via small bounded edits — it never re-types a link.
    """
    duplicate_count = sum(len(c) for c in clusters)

    lines = [
        "# MLflow Security Advisory Dedupe",
        "",
        f"**Analyzed {total} advisory(ies) (state: {state}).**",
        "",
        "> **Notify-only.** These are similarity-based candidates (threshold "
        f"{threshold}) for the maintainer to confirm. No advisory has been or will "
        "be modified by this analysis.",
        "",
        "## Confidence tiers",
        "",
        "<!-- Fill in after enriching the clusters below. Reference clusters by "
        "number; do NOT move the cluster blocks. -->",
        "- **Strong duplicates (high confidence):** TODO",
        "- **Medium-confidence pairs:** TODO",
    ]

    for n, cluster in enumerate(clusters, start=1):
        original, *duplicates = cluster
        lines.append("")
        lines.append(
            f"### Cluster {n} ({len(cluster)} reports) — "
            "<!-- TITLE: TODO --> — <!-- CONFIDENCE: TODO -->"
        )
        lines.append("")
        lines.append("**Original source (earliest report):**")
        lines.append(format_member_markdown(original, original=True))
        lines.append("")
        lines.append("**Later duplicates:**")
        lines.extend(format_member_markdown(d) for d in duplicates)
        lines.append("")
        lines.append("> _Reasoning: TODO_")

    lines.append("")
    lines.append(f"## Not part of any duplicate cluster — review independently ({len(singletons)})")
    lines.append("")
    if singletons:
        lines.append("No duplicate candidates detected for these; each needs an independent look.")
        lines.extend(format_member_markdown(a) for a in singletons)
    else:
        lines.append("None - every analyzed advisory fell into a candidate cluster.")

    lines.append("")
    lines.append(
        f"_{duplicate_count} advisory(ies) in {len(clusters)} cluster(s); "
        f"{len(singletons)} not part of any cluster._"
    )
    return "\n".join(lines)


def format_output(
    clusters: list[list[SecurityAdvisory]],
    total: int,
    threshold: float,
    state: str,
    singletons: list[SecurityAdvisory],
) -> str:
    duplicate_count = sum(len(c) for c in clusters)

    lines = [
        f"# Candidate duplicate advisories ({len(clusters)} cluster(s), threshold {threshold})",
        "",
        f"Analyzed {total} advisory(ies) (state: {state}).",
        "",
        "Notify-only: these are similarity-based candidates for the maintainer to confirm. "
        "No advisory has been or will be modified.",
    ]

    for n, cluster in enumerate(clusters, start=1):
        original, *duplicates = cluster
        lines.append("")
        lines.append(f"## Cluster {n} ({len(cluster)} reports)")
        lines.append("")
        lines.append("ORIGINAL SOURCE (earliest report):")
        lines.append(format_member(original))
        lines.append("")
        lines.append("LATER DUPLICATES:")
        lines.extend(format_member(d) for d in duplicates)

    lines.extend(format_singletons(singletons))

    lines.append("")
    lines.append(
        f"{duplicate_count} advisory(ies) in {len(clusters)} cluster(s); "
        f"{len(singletons)} not part of any cluster (listed above)."
    )
    return "\n".join(lines)


@dataclass
class DedupeResult:
    clusters: list[list[SecurityAdvisory]]
    singletons: list[SecurityAdvisory]
    total: int


async def dedupe_advisories(
    repo: str, state: str, threshold: float, min_cluster_size: int
) -> DedupeResult:
    owner, name = parse_repo(repo)
    api_state = None if state == "all" else state
    log(f"Fetching security advisories for {owner}/{name} (state={state})")
    async with GitHubClient() as client:
        advisories = [a async for a in client.get_security_advisories(owner, name, api_state)]
    log(f"Found {len(advisories)} advisory(ies); clustering")
    clusters = cluster_advisories(advisories, threshold, min_cluster_size)
    clustered = {a.ghsa_id for cluster in clusters for a in cluster}
    singletons = sorted((a for a in advisories if a.ghsa_id not in clustered), key=sort_key)
    return DedupeResult(clusters=clusters, singletons=singletons, total=len(advisories))


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "dedupe-advisories", help="Cluster likely-duplicate security advisories"
    )
    parser.add_argument(
        "--repo", default="mlflow/mlflow", help="owner/repo (default: mlflow/mlflow)"
    )
    parser.add_argument(
        "--state",
        choices=[*STATES, "all"],
        default="triage",
        help="Advisory state to compare (default: triage)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for candidate duplicates (default: 0.75)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum reports per cluster to report (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Write a fully-rendered Markdown report scaffold to this path (with "
            "linked GHSA ids and TODO markers for the model to enrich). When "
            "omitted, the plain-text report is printed to stdout instead."
        ),
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    result = asyncio.run(
        dedupe_advisories(args.repo, args.state, args.threshold, args.min_cluster_size)
    )
    if args.output is not None:
        scaffold = format_scaffold(
            result.clusters, result.total, args.threshold, args.state, result.singletons
        )
        args.output.write_text(scaffold, encoding="utf-8")
        log(f"Wrote scaffold to {args.output}")
        print(
            f"Wrote report scaffold to {args.output} "
            f"({len(result.clusters)} cluster(s); {len(result.singletons)} singleton(s); "
            f"{result.total} analyzed). Enrich the TODO markers in place."
        )
    else:
        report = format_output(
            result.clusters, result.total, args.threshold, args.state, result.singletons
        )
        print(report)
