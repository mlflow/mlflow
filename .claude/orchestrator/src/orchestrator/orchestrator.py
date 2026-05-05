"""Main orchestration flow.

The flow per `/review` invocation:

1. Fetch PR metadata, diff, and changed-file contents.
2. If the head SHA was already reviewed and `--no-cache` was not passed, exit.
3. Load agent prompts.
4. Run agents:
     - Lite mode: only mlflow-reviewer in discovery mode.
     - Default mode: spotter + reviewer-standalone in parallel; then reviewer
       in opinion mode on the spotter findings; then cluster the standalone
       and opinion outputs.
5. Apply dedup rules against existing review threads.
6. Output: dry-run prints JSON to stdout; live mode (Stack 3) posts comments.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import sys
from dataclasses import dataclass
from enum import Enum

from orchestrator import agents
from orchestrator.anthropic_client import (
    AnthropicClient,
    AnthropicConfig,
    Message,
    ModelChoice,
)
from orchestrator.cluster import (
    Cluster,
    SourceFinding,
    clusters_to_drafts,
    materialize_clusters,
)
from orchestrator.dedup import DraftFinding, FilterDecision, filter_findings
from orchestrator.github_client import GitHubClient, PRMetadata
from orchestrator.judges import (
    cluster_judge,
    make_semantic_judge,
    reviewer_discovery_user_message,
    reviewer_opinion_user_message,
    spotter_user_message,
)
from orchestrator.posting import (
    already_reviewed_at_sha,
    post_inline_comment,
    post_summary_comment,
)

_logger = logging.getLogger(__name__)


class Mode(str, Enum):
    DEFAULT = "default"
    LITE = "lite"


@dataclass(frozen=True)
class Config:
    pr_number: int
    mode: Mode = Mode.DEFAULT
    no_cache: bool = False
    hybrid: bool = False
    dry_run: bool = True
    repo: str = "mlflow/mlflow"
    bot_login: str = "mlflow-reviewer[bot]"

    @property
    def models(self) -> ModelChoice:
        return ModelChoice.hybrid() if self.hybrid else ModelChoice.default()


@dataclass(frozen=True)
class ReviewOutput:
    pr_number: int
    head_sha: str
    mode: Mode
    posted: list[FilterDecision]
    skipped: list[FilterDecision]
    summary: str


def parse_reviewer_discovery_findings(text: str) -> list[SourceFinding]:
    data = json.loads(text)
    return [
        SourceFinding(
            source="reviewer_standalone",
            path=f.get("path") or "",
            line=int(f.get("line") or 0),
            body=f.get("body") or "",
            severity=f.get("risk_to_severity") or _risk_to_severity(f.get("risk")),
            rule_id=f.get("rule_id"),
        )
        for f in data.get("findings", [])
    ]


def parse_spotter_findings(text: str) -> list[dict[str, object]]:
    data = json.loads(text)
    return data.get("findings", [])


def parse_reviewer_opinion_raises(
    text: str, spotter_findings: list[dict[str, object]]
) -> list[SourceFinding]:
    """Extract `decision == "raise"` opinions and join with spotter findings."""
    data = json.loads(text)
    by_id = {f["finding_id"]: f for f in spotter_findings}
    out: list[SourceFinding] = []
    for op in data.get("opinions", []):
        if op.get("decision") != "raise":
            continue
        spotter = by_id.get(op["finding_id"], {})
        out.append(
            SourceFinding(
                source="reviewer_opinion",
                path=spotter.get("path") or "",
                line=int(spotter.get("line") or 0),
                body=op.get("voice_rewrite") or spotter.get("body") or "",
                severity=op.get("severity") or "ask",
                rule_id=op.get("rule_id"),
            )
        )
    return out


_RISK_TO_SEVERITY = {
    "No risk": "nit",
    "Low risk": "ask",
    "Needs discussion": "firm",
}


def _risk_to_severity(risk: str | None) -> str:
    if risk is None:
        return "ask"
    return _RISK_TO_SEVERITY.get(risk, "ask")


async def _run_reviewer_standalone(
    client: AnthropicClient,
    pr: PRMetadata,
    diff: str,
    files: dict[str, str],
    models: ModelChoice,
) -> list[SourceFinding]:
    result = await client.complete(
        role=models.reviewer_standalone,
        system=agents.reviewer_prompt(),
        messages=[
            Message(
                role="user",
                content=reviewer_discovery_user_message(pr.number, pr.title, diff, files),
            )
        ],
    )
    return parse_reviewer_discovery_findings(result.text)


async def _run_spotter(
    client: AnthropicClient,
    pr: PRMetadata,
    diff: str,
    files: dict[str, str],
    models: ModelChoice,
) -> list[dict[str, object]]:
    result = await client.complete(
        role=models.spotter,
        system=agents.spotter_prompt(),
        messages=[
            Message(
                role="user",
                content=spotter_user_message(pr.number, pr.title, diff, files),
            )
        ],
    )
    return parse_spotter_findings(result.text)


async def _run_reviewer_opinion(
    client: AnthropicClient,
    pr: PRMetadata,
    diff: str,
    spotter_findings: list[dict[str, object]],
    models: ModelChoice,
) -> list[SourceFinding]:
    if not spotter_findings:
        return []
    result = await client.complete(
        role=models.reviewer_opinion,
        system=agents.reviewer_prompt(),
        messages=[
            Message(
                role="user",
                content=reviewer_opinion_user_message(
                    pr.number, pr.title, diff, json.dumps(spotter_findings)
                ),
            )
        ],
    )
    return parse_reviewer_opinion_raises(result.text, spotter_findings)


async def _cluster(
    client: AnthropicClient,
    list_a: list[SourceFinding],
    list_b: list[SourceFinding],
    models: ModelChoice,
) -> list[Cluster]:
    if not list_a and not list_b:
        return []
    if not list_a:
        return [Cluster(members=[f]) for f in list_b]
    if not list_b:
        return [Cluster(members=[f]) for f in list_a]
    judge_text = await cluster_judge(client, list_a, list_b, models=models)
    return materialize_clusters(judge_text, list_a, list_b)


def _read_files_at_head(
    github: GitHubClient, paths: tuple[str, ...], head_sha: str
) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in paths:
        try:
            files[path] = github.read_file_at_head(path, head_sha)
        except Exception as e:
            _logger.warning("Could not read %s at %s: %s", path, head_sha, e)
    return files


async def run_review(
    config: Config,
    *,
    anthropic: AnthropicConfig | None = None,
) -> ReviewOutput:
    anthropic = anthropic or AnthropicConfig.from_env()
    client = AnthropicClient(anthropic)
    github = GitHubClient(repo=config.repo, bot_login=config.bot_login)

    pr = github.get_pr_metadata(config.pr_number)

    if not config.no_cache and already_reviewed_at_sha(
        config.repo, config.pr_number, pr.head_sha, bot_login=config.bot_login
    ):
        summary = (
            f"PR #{pr.number} already reviewed at head_sha={pr.head_sha}; "
            "skipping. Use --no-cache to force a fresh run."
        )
        _logger.info(summary)
        return ReviewOutput(
            pr_number=pr.number,
            head_sha=pr.head_sha,
            mode=config.mode,
            posted=[],
            skipped=[],
            summary=summary,
        )

    diff = github.get_pr_diff(config.pr_number)
    files = _read_files_at_head(github, pr.changed_paths, pr.head_sha)
    threads = github.get_review_threads(config.pr_number, pr_author=pr.author_login)

    if config.mode == Mode.LITE:
        standalone = await _run_reviewer_standalone(client, pr, diff, files, config.models)
        clusters = [Cluster(members=[f]) for f in standalone]
    else:
        standalone, spotter_findings = await asyncio.gather(
            _run_reviewer_standalone(client, pr, diff, files, config.models),
            _run_spotter(client, pr, diff, files, config.models),
        )
        opinion = await _run_reviewer_opinion(client, pr, diff, spotter_findings, config.models)
        clusters = await _cluster(client, standalone, opinion, config.models)

    drafts = clusters_to_drafts(clusters)

    semantic_judge = make_semantic_judge(client, config.models)
    decisions = await filter_findings(
        drafts,
        threads,
        pr_author_login=pr.author_login,
        semantic_judge=semantic_judge,
        thread_bodies_by_thread={},
    )
    posted = [d for d in decisions if d.posted]
    skipped = [d for d in decisions if not d.posted]

    summary = (
        f"PR #{pr.number}: drafts={len(drafts)}, posted={len(posted)}, "
        f"skipped={len(skipped)}, mode={config.mode.value}, head_sha={pr.head_sha}"
    )
    _logger.info(summary)

    if config.dry_run:
        _emit_dry_run_json(pr, posted, skipped, summary)
    else:
        _post_findings(config.repo, pr, posted, skipped, config.mode.value)

    return ReviewOutput(
        pr_number=pr.number,
        head_sha=pr.head_sha,
        mode=config.mode,
        posted=posted,
        skipped=skipped,
        summary=summary,
    )


def _post_findings(
    repo: str,
    pr: PRMetadata,
    posted: list[FilterDecision],
    skipped: list[FilterDecision],
    mode: str,
) -> None:
    for decision in posted:
        try:
            comment = post_inline_comment(repo, pr.number, pr.head_sha, decision.finding)
            _logger.info("Posted inline: %s", comment.html_url)
        except Exception:
            _logger.exception(
                "Failed to post inline comment for %s:%s",
                decision.finding.path,
                decision.finding.line,
            )
    summary_comment = post_summary_comment(
        repo,
        pr,
        posted_count=len(posted),
        skipped_count=len(skipped),
        mode=mode,
    )
    _logger.info("Posted summary: %s", summary_comment.html_url)


def _emit_dry_run_json(
    pr: PRMetadata,
    posted: list[FilterDecision],
    skipped: list[FilterDecision],
    summary: str,
) -> None:
    payload = {
        "pr": pr.number,
        "head_sha": pr.head_sha,
        "summary": summary,
        "posted": [{**dataclasses.asdict(d.finding), "skip_reason": None} for d in posted],
        "skipped": [
            {**dataclasses.asdict(d.finding), "skip_reason": d.skip_reason} for d in skipped
        ],
    }
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    sys.stdout.flush()


__all__ = [
    "Config",
    "DraftFinding",
    "Mode",
    "ReviewOutput",
    "run_review",
]
