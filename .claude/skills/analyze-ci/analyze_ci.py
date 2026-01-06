"""
Fetch and analyze logs from failed GitHub Action jobs.

Usage:
    uv run .claude/skills/analyze-ci/analyze_ci.py <pr_url> [pr_url ...]
    uv run .claude/skills/analyze-ci/analyze_ci.py <job_url> [job_url ...]
"""
# /// script
# dependencies = [
#   "aiohttp",
#   "claude-agent-sdk",
#   "tiktoken",
#   "typing_extensions",
# ]
# ///
# ruff: noqa: T201

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import aiohttp
import tiktoken
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from typing_extensions import Self

MAX_LOG_TOKENS = 100_000


@dataclass
class JobLogs:
    workflow_name: str
    job_name: str
    job_url: str
    failed_step: str | None
    logs: str


@dataclass
class AnalysisResult:
    text: str
    total_cost_usd: float | None
    usage: dict[str, Any] | None


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def get_github_token(cli_token: str | None = None) -> str:
    token = cli_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        try:
            token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
            log("Using token from 'gh auth token'")
        except (subprocess.CalledProcessError, FileNotFoundError):
            log("Error: GITHUB_TOKEN is required (--github-token, env var, or 'gh auth token')")
            sys.exit(1)
    return token


class GitHubClient:
    """Async GitHub API client."""

    def __init__(self, token: str, base_url: str = "https://api.github.com") -> None:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._session = aiohttp.ClientSession(base_url=base_url, headers=headers)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._session.close()

    async def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        async with self._session.get(endpoint, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def get_raw(self, endpoint: str) -> aiohttp.ClientResponse:
        return await self._session.get(endpoint, allow_redirects=True)

    async def paginate(
        self,
        endpoint: str,
        key: str,
        params: dict[str, Any] | None = None,
        per_page: int = 100,
    ) -> AsyncIterator[dict[str, Any]]:
        page = 1
        params = (params or {}) | {"per_page": per_page}

        while True:
            params["page"] = page
            result = await self.get(endpoint, params)
            items = result.get(key, [])
            if not items:
                break

            for item in items:
                yield item

            if len(items) < per_page:
                break
            page += 1

    async def get_pr_details(self, repo: str, pr_number: int) -> dict[str, Any]:
        return await self.get(f"/repos/{repo}/pulls/{pr_number}")

    async def get_workflow_runs(
        self,
        repo: str,
        head_sha: str | None = None,
        status: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        params: dict[str, Any] = {}
        if head_sha:
            params["head_sha"] = head_sha
        if status:
            params["status"] = status
        async for run in self.paginate(f"/repos/{repo}/actions/runs", "workflow_runs", params):
            yield run

    async def get_jobs(self, repo: str, run_id: int) -> AsyncIterator[dict[str, Any]]:
        endpoint = f"/repos/{repo}/actions/runs/{run_id}/jobs"
        async for job in self.paginate(endpoint, "jobs"):
            yield job

    async def get_job_details(self, repo: str, job_id: int) -> dict[str, Any]:
        return await self.get(f"/repos/{repo}/actions/jobs/{job_id}")

    async def get_run_details(self, repo: str, run_id: int) -> dict[str, Any]:
        return await self.get(f"/repos/{repo}/actions/runs/{run_id}")


TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ?")


def to_seconds(ts: str) -> str:
    """Truncate timestamp to seconds precision for comparison."""
    return ts[:19]  # "2026-01-05T07:17:56.1234567Z" -> "2026-01-05T07:17:56"


async def iter_job_logs(
    client: GitHubClient,
    repo: str,
    job_id: int,
    started_at: str,
    completed_at: str,
) -> AsyncIterator[str]:
    """Yield log lines filtered by time range."""
    async with await client.get_raw(f"/repos/{repo}/actions/jobs/{job_id}/logs") as response:
        response.raise_for_status()

        # ISO 8601 timestamps are lexicographically sortable, so we can compare as strings
        start_secs = to_seconds(started_at)
        end_secs = to_seconds(completed_at)
        in_range = False

        async for line in response.content:
            line = line.decode("utf-8").rstrip("\r\n")
            if TIMESTAMP_PATTERN.match(line):
                ts_secs = to_seconds(line)
                if ts_secs > end_secs:
                    return  # Past end time, stop reading
                in_range = ts_secs >= start_secs
                line = line.split(" ", 1)[1]  # strip timestamp
            if in_range:
                yield line


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
PYTEST_SECTION_PATTERN = re.compile(r"^={6,} (.+?) ={6,}$")

# Pytest sections to skip (not useful for failure analysis)
PYTEST_SKIP_SECTIONS = {
    "test session starts",
    "warnings summary",
    "per-file durations",
    "remaining threads",
}


async def compact_logs(lines: AsyncIterator[str]) -> str:
    """Clean logs: strip timestamps, ANSI colors, and filter noisy pytest sections."""
    result: list[str] = []
    skip_section = False

    async for line in lines:
        line = ANSI_PATTERN.sub("", line)
        if match := PYTEST_SECTION_PATTERN.match(line.strip()):
            section_name = match.group(1).strip().lower()
            skip_section = any(name in section_name for name in PYTEST_SKIP_SECTIONS)
        if not skip_section:
            result.append(line)

    logs = "\n".join(result)
    tokens = tiktoken.get_encoding("p50k_base").encode(logs)
    log(f"Compacted logs: {len(tokens):,} tokens")
    return logs


def truncate_logs(logs: str, max_tokens: int = MAX_LOG_TOKENS) -> str:
    """Truncate logs to fit within token limit, keeping the end (where errors are)."""
    # Note: tiktoken token count is an estimation and may differ slightly from
    # the official token count API
    tokenizer = tiktoken.get_encoding("p50k_base")
    tokens = tokenizer.encode(logs)
    if len(tokens) <= max_tokens:
        return logs
    log(f"Truncating logs from {len(tokens):,} to {max_tokens:,} tokens")
    truncated = tokenizer.decode(tokens[-max_tokens:])
    return f"(showing last {max_tokens:,} tokens)\n{truncated}"


def get_failed_step(job_details: dict[str, Any]) -> dict[str, Any] | None:
    steps = job_details.get("steps", [])
    for step in steps:
        if step.get("conclusion") == "failure":
            return step
    return None


PR_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/pull/(\d+)")
JOB_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/actions/runs/(\d+)/job/(\d+)")


@dataclass
class JobRun:
    repo: str
    run_id: int
    job_id: int

    @property
    def job_url(self) -> str:
        return f"https://github.com/{self.repo}/actions/runs/{self.run_id}/job/{self.job_id}"


async def get_failed_jobs_from_pr(client: GitHubClient, repo: str, pr_number: int) -> list[JobRun]:
    log(f"Fetching https://github.com/{repo}/pull/{pr_number}")
    pr_details = await client.get_pr_details(repo, pr_number)
    head_sha = pr_details["head"]["sha"]
    log(f"PR head SHA: {head_sha[:8]}")

    runs = client.get_workflow_runs(repo, head_sha)
    failed_runs = [r async for r in runs if r.get("conclusion") == "failure"]
    log(f"Found {len(failed_runs)} failed workflow run(s)")

    async def failed_jobs(run_id: int) -> list[dict[str, Any]]:
        return [j async for j in client.get_jobs(repo, run_id) if j.get("conclusion") == "failure"]

    failed_jobs_results = await asyncio.gather(*[failed_jobs(run["id"]) for run in failed_runs])

    run_job_pairs = [
        (run, job) for run, jobs in zip(failed_runs, failed_jobs_results) for job in jobs
    ]
    log(f"Found {len(run_job_pairs)} failed job(s)")

    return [JobRun(repo, run["id"], job["id"]) for run, job in run_job_pairs]


async def resolve_urls(client: GitHubClient, urls: list[str]) -> list[JobRun]:
    job_runs: list[JobRun] = []

    for url in urls:
        if match := JOB_URL_PATTERN.search(url):
            job_runs.append(JobRun(match.group(1), int(match.group(2)), int(match.group(3))))
        elif match := PR_URL_PATTERN.search(url):
            jobs = await get_failed_jobs_from_pr(client, match.group(1), int(match.group(2)))
            job_runs.extend(jobs)
        else:
            log(f"Error: Invalid URL: {url}")
            log("Expected PR URL (github.com/owner/repo/pull/123)")
            log("Or job URL (github.com/owner/repo/actions/runs/123/job/456)")
            sys.exit(1)

    return job_runs


async def fetch_single_job_logs(client: GitHubClient, job: JobRun) -> JobLogs:
    log(f"Fetching job {job.job_id} from {job.repo}")

    job_details = await client.get_job_details(job.repo, job.job_id)
    run_details = await client.get_run_details(job.repo, job.run_id)

    workflow_name = run_details.get("name", "Unknown workflow")
    job_name = job_details.get("name", "Unknown job")
    failed_step = get_failed_step(job_details)
    if not failed_step:
        raise ValueError(f"No failed step found for job {job.job_id}")

    log(f"Fetching logs for '{workflow_name} / {job_name}'")
    cleaned_logs = await compact_logs(
        iter_job_logs(
            client,
            job.repo,
            job.job_id,
            started_at=failed_step["started_at"],
            completed_at=failed_step["completed_at"],
        )
    )
    truncated_logs = truncate_logs(cleaned_logs)
    failed_step_name = failed_step.get("name")

    return JobLogs(
        workflow_name=workflow_name,
        job_name=job_name,
        job_url=job.job_url,
        failed_step=failed_step_name,
        logs=truncated_logs,
    )


ANALYZE_SYSTEM_PROMPT = """\
You are a CI failure analyzer. Analyze the provided CI logs and produce a concise failure summary.

Instructions:
1. Identify the root cause of each failure
2. Extract specific error messages (assertion errors, exceptions, stack traces)
3. For pytest failures, include full test names (e.g., tests/test_foo.py::test_bar)
4. Include relevant log snippets showing error context

Output format for each failed job:
```
Failed job: <workflow name> / <job name>
Failed step: <step name>
URL: <job_url>

<1-2 paragraph summary with root cause, error messages, test names, and key log snippets>
```
"""


def format_single_job_for_analysis(job: JobLogs) -> str:
    parts = [
        f"## {job.workflow_name} / {job.job_name}",
        f"URL: {job.job_url}",
        f"Failed step: {job.failed_step or 'Unknown'}",
        "",
        "```",
        job.logs,
        "```",
    ]
    return "\n".join(parts)


async def analyze_single_job(job: JobLogs) -> AnalysisResult:
    formatted_logs = format_single_job_for_analysis(job)
    prompt = f"Analyze this CI failure:\n\n{formatted_logs}"

    options = ClaudeAgentOptions(
        system_prompt=ANALYZE_SYSTEM_PROMPT,
        max_turns=1,
        model="haiku",
    )

    text_parts: list[str] = []
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None

    async for message in query(prompt=prompt, options=options):
        match message:
            case AssistantMessage(content=content):
                for block in content:
                    match block:
                        case TextBlock(text=text):
                            text_parts.append(text)
            case ResultMessage(total_cost_usd=cost, usage=u):
                total_cost_usd = cost
                usage = u

    return AnalysisResult(
        text="".join(text_parts),
        total_cost_usd=total_cost_usd,
        usage=usage,
    )


def format_result(result: AnalysisResult, debug: bool = False) -> str:
    """Format analysis result, optionally with usage JSON."""
    if not debug:
        return result.text

    filtered_usage = None
    if result.usage:
        filtered_usage = {k: v for k, v in result.usage.items() if "tokens" in k}
    usage_data = {"total_cost_usd": result.total_cost_usd, "usage": filtered_usage}
    usage_json = json.dumps(usage_data, indent=2)
    return f"{result.text}\n\n```json\n{usage_json}\n```"


async def analyze_with_claude(jobs: list[JobLogs], debug: bool = False) -> str:
    """Analyze each job in parallel to speed up processing."""
    log(f"Analyzing {len(jobs)} job(s) in parallel...")
    results = await asyncio.gather(*[analyze_single_job(job) for job in jobs])

    separator = "\n\n---\n\n"
    return separator.join(format_result(r, debug) for r in results)


async def cmd_analyze_async(urls: list[str], github_token: str, debug: bool = False) -> None:
    async with GitHubClient(github_token) as client:
        # Resolve URLs to job targets
        jobs = await resolve_urls(client, urls)

        if not jobs:
            log("No failed jobs found")
            return

        # Fetch logs for all jobs
        log(f"Fetching logs for {len(jobs)} job(s)")
        results = await asyncio.gather(*[fetch_single_job_logs(client, job) for job in jobs])

    # Analyze with Claude
    log("Analyzing logs with Claude...")
    summary = await analyze_with_claude(results, debug)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and analyze logs from failed GitHub Action jobs"
    )
    parser.add_argument("urls", nargs="+", help="PR URL or job URL(s) to analyze")
    parser.add_argument("--github-token", help="GitHub token (or set GITHUB_TOKEN)")
    parser.add_argument("--debug", action="store_true", help="Show token and cost info")

    args = parser.parse_args()
    github_token = get_github_token(args.github_token)

    asyncio.run(cmd_analyze_async(args.urls, github_token, args.debug))


if __name__ == "__main__":
    main()
