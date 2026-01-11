# ruff: noqa: T201
"""Analyze failed GitHub Action jobs."""

import argparse
import asyncio
import json
import re
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import tiktoken
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

from skills.github import GitHubClient, Job, JobStep, get_github_token

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


TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ?")


def to_seconds(ts: str) -> str:
    """Truncate timestamp to seconds precision for comparison."""
    return ts[:19]  # "2026-01-05T07:17:56.1234567Z" -> "2026-01-05T07:17:56"


async def iter_job_logs(
    client: GitHubClient,
    job: Job,
    failed_step: JobStep,
) -> AsyncIterator[str]:
    """Yield log lines filtered by time range."""
    if not failed_step.started_at or not failed_step.completed_at:
        raise ValueError(f"Failed step missing timestamps for job {job.id}")

    async with await client.get_raw(f"{job.url}/logs") as response:
        response.raise_for_status()

        # ISO 8601 timestamps are lexicographically sortable, so we can compare as strings
        start_secs = to_seconds(failed_step.started_at)
        end_secs = to_seconds(failed_step.completed_at)
        in_range = False

        async for line_bytes in response.content:
            line_str = line_bytes.decode("utf-8").rstrip("\r\n")
            if TIMESTAMP_PATTERN.match(line_str):
                ts_secs = to_seconds(line_str)
                if ts_secs > end_secs:
                    return  # Past end time, stop reading
                in_range = ts_secs >= start_secs
                line_str = line_str.split(" ", 1)[1]  # strip timestamp
            if in_range:
                yield line_str


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


PR_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/pull/(\d+)")
JOB_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/actions/runs/(\d+)/job/(\d+)")


async def get_failed_jobs_from_pr(
    client: GitHubClient, owner: str, repo: str, pr_number: int
) -> list[Job]:
    log(f"Fetching https://github.com/{owner}/{repo}/pull/{pr_number}")
    pr = await client.get_pr(owner, repo, pr_number)
    head_sha = pr.head.sha
    log(f"PR head SHA: {head_sha[:8]}")

    runs = client.get_workflow_runs(owner, repo, head_sha)
    failed_runs = [r async for r in runs if r.conclusion == "failure"]
    log(f"Found {len(failed_runs)} failed workflow run(s)")

    async def failed_jobs(run_id: int) -> list[Job]:
        return [j async for j in client.get_jobs(owner, repo, run_id) if j.conclusion == "failure"]

    failed_jobs_results = await asyncio.gather(*[failed_jobs(run.id) for run in failed_runs])

    jobs = [job for jobs in failed_jobs_results for job in jobs]
    log(f"Found {len(jobs)} failed job(s)")

    return jobs


async def resolve_urls(client: GitHubClient, urls: list[str]) -> list[Job]:
    jobs: list[Job] = []

    for url in urls:
        if match := JOB_URL_PATTERN.search(url):
            repo_full = match.group(1)
            owner, repo = repo_full.split("/")
            job_id = int(match.group(3))
            job = await client.get_job(owner, repo, job_id)
            jobs.append(job)
        elif match := PR_URL_PATTERN.search(url):
            repo_full = match.group(1)
            owner, repo = repo_full.split("/")
            pr_jobs = await get_failed_jobs_from_pr(client, owner, repo, int(match.group(2)))
            jobs.extend(pr_jobs)
        else:
            log(f"Error: Invalid URL: {url}")
            log("Expected PR URL (github.com/owner/repo/pull/123)")
            log("Or job URL (github.com/owner/repo/actions/runs/123/job/456)")
            sys.exit(1)

    return jobs


async def fetch_single_job_logs(client: GitHubClient, job: Job) -> JobLogs:
    log(f"Fetching logs for '{job.workflow_name} / {job.name}'")

    failed_step = next((s for s in job.steps if s.conclusion == "failure"), None)
    if not failed_step:
        raise ValueError(f"No failed step found for job {job.id}")
    cleaned_logs = await compact_logs(iter_job_logs(client, job, failed_step))
    truncated_logs = truncate_logs(cleaned_logs)
    failed_step_name = failed_step.name

    return JobLogs(
        workflow_name=job.workflow_name,
        job_name=job.name,
        job_url=job.html_url,
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


async def analyze_jobs(jobs: list[JobLogs], debug: bool = False) -> str:
    """Analyze each job in parallel to speed up processing."""
    log(f"Analyzing {len(jobs)} job(s) in parallel...")
    results = await asyncio.gather(*[analyze_single_job(job) for job in jobs])

    separator = "\n\n---\n\n"
    return separator.join(format_result(r, debug) for r in results)


async def cmd_analyze_async(urls: list[str], debug: bool = False) -> None:
    github_token = get_github_token()
    async with GitHubClient(github_token) as client:
        # Resolve URLs to job targets
        jobs = await resolve_urls(client, urls)

        if not jobs:
            log("No failed jobs found")
            return

        # Fetch logs for all jobs
        log(f"Fetching logs for {len(jobs)} job(s)")
        results = await asyncio.gather(*[fetch_single_job_logs(client, job) for job in jobs])

    # Analyze logs
    log("Analyzing logs...")
    summary = await analyze_jobs(results, debug)
    print(summary)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("analyze-ci", help="Analyze failed CI jobs")
    parser.add_argument("urls", nargs="+", help="PR URL or job URL(s)")
    parser.add_argument("--debug", action="store_true", help="Show token/cost info")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(cmd_analyze_async(args.urls, args.debug))
