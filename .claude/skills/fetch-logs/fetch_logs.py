"""
Fetch and analyze logs from failed GitHub Action jobs.

Usage:
    # Analyze CI failures (fetch logs and summarize with Claude)
    uv run fetch_logs.py <pr_url>
    uv run fetch_logs.py <job_url> [job_url ...]
"""
# /// script
# dependencies = [
#     "aiohttp",
#     "tiktoken",
#     "claude-agent-sdk",
# ]
# ///
# ruff: noqa: T201

import argparse
import asyncio
import os
import re
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import aiohttp
import tiktoken

GITHUB_API_BASE = "https://api.github.com"
MAX_LOG_TOKENS = 100_000


@dataclass
class JobLogs:
    workflow_name: str
    job_name: str
    job_url: str
    failed_step: str | None
    logs: str


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


def get_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def api_get(
    session: aiohttp.ClientSession,
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    url = f"{GITHUB_API_BASE}{endpoint}"
    async with session.get(url, params=params) as response:
        response.raise_for_status()
        return await response.json()


async def get_pr_details(
    session: aiohttp.ClientSession, pr_number: int, repo: str
) -> dict[str, Any]:
    return await api_get(session, f"/repos/{repo}/pulls/{pr_number}")


async def get_workflow_runs(
    session: aiohttp.ClientSession,
    repo: str,
    head_sha: str,
    status: str = "completed",
) -> AsyncIterator[dict[str, Any]]:
    page = 1
    per_page = 100

    while True:
        params = {
            "head_sha": head_sha,
            "status": status,
            "per_page": per_page,
            "page": page,
        }
        result = await api_get(session, f"/repos/{repo}/actions/runs", params)
        runs = result.get("workflow_runs", [])
        if not runs:
            break

        for run in runs:
            yield run
        page += 1

        if len(runs) < per_page:
            break


async def get_failed_jobs(
    session: aiohttp.ClientSession, run_id: int, repo: str
) -> list[dict[str, Any]]:
    all_jobs: list[dict[str, Any]] = []
    page = 1
    per_page = 100

    while True:
        params = {"per_page": per_page, "page": page}
        result = await api_get(session, f"/repos/{repo}/actions/runs/{run_id}/jobs", params)
        jobs = result.get("jobs", [])
        if not jobs:
            break

        all_jobs.extend(jobs)
        page += 1

        if len(jobs) < per_page:
            break

    return [job for job in all_jobs if job.get("conclusion") == "failure"]


TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ?")


def to_seconds(ts: str) -> str:
    """Truncate timestamp to seconds precision for comparison."""
    return ts[:19]  # "2026-01-05T07:17:56.1234567Z" -> "2026-01-05T07:17:56"


async def iter_job_logs(
    session: aiohttp.ClientSession,
    job_id: int,
    repo: str,
    started_at: str,
    completed_at: str,
):
    """Yield log lines filtered by time range."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/actions/jobs/{job_id}/logs"
    async with session.get(url, allow_redirects=True) as response:
        response.raise_for_status()

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
    """Get the first failed step from job details."""
    steps = job_details.get("steps", [])
    for step in steps:
        if step.get("conclusion") == "failure":
            return step
    return None


JOB_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/actions/runs/(\d+)/job/(\d+)")
PR_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/pull/(\d+)")


def parse_job_url(url: str) -> tuple[str, int, int]:
    """Parse a GitHub Actions job URL and return (repo, run_id, job_id)."""
    if not (match := JOB_URL_PATTERN.search(url)):
        raise ValueError(f"Invalid job URL: {url}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def parse_pr_url(url: str) -> tuple[str, int] | None:
    """Parse a GitHub PR URL and return (repo, pr_number), or None if not a PR URL."""
    if match := PR_URL_PATTERN.search(url):
        return match.group(1), int(match.group(2))
    return None


# ============================================================================
# Helpers for fetching job logs
# ============================================================================


async def get_job_details(session: aiohttp.ClientSession, repo: str, job_id: int) -> dict[str, Any]:
    return await api_get(session, f"/repos/{repo}/actions/jobs/{job_id}")


async def get_run_details(session: aiohttp.ClientSession, repo: str, run_id: int) -> dict[str, Any]:
    return await api_get(session, f"/repos/{repo}/actions/runs/{run_id}")


async def fetch_single_job_logs(
    session: aiohttp.ClientSession,
    job_url: str,
) -> JobLogs:
    repo, run_id, job_id = parse_job_url(job_url)
    log(f"Fetching job {job_id} from {repo}")

    job_details = await get_job_details(session, repo, job_id)
    run_details = await get_run_details(session, repo, run_id)

    workflow_name = run_details.get("name", "Unknown workflow")
    job_name = job_details.get("name", "Unknown job")
    failed_step = get_failed_step(job_details)

    log(f"Fetching logs for '{workflow_name} / {job_name}'")
    cleaned_logs = await compact_logs(
        iter_job_logs(
            session,
            job_id,
            repo,
            started_at=failed_step["started_at"],
            completed_at=failed_step["completed_at"],
        )
    )
    truncated_logs = truncate_logs(cleaned_logs)
    failed_step_name = failed_step.get("name")

    return JobLogs(
        workflow_name=workflow_name,
        job_name=job_name,
        job_url=job_url,
        failed_step=failed_step_name,
        logs=truncated_logs,
    )


# ============================================================================
# Subcommand: analyze
# ============================================================================

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
    """Format a single job's logs for Claude analysis."""
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


async def analyze_single_job(job: JobLogs) -> str:
    """Analyze a single job's logs with Claude."""
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

    formatted_logs = format_single_job_for_analysis(job)
    prompt = f"Analyze this CI failure:\n\n{formatted_logs}"

    options = ClaudeAgentOptions(
        system_prompt=ANALYZE_SYSTEM_PROMPT,
        max_turns=1,
        model="haiku",
    )

    result: list[str] = []
    async for message in query(prompt=prompt, options=options):
        match message:
            case AssistantMessage(content=content):
                for block in content:
                    match block:
                        case TextBlock(text=text):
                            result.append(text)
    return "".join(result)


async def analyze_with_claude(jobs: list[JobLogs]) -> str:
    """Analyze each job in parallel to speed up processing."""
    log(f"Analyzing {len(jobs)} job(s) in parallel...")
    results = await asyncio.gather(*[analyze_single_job(job) for job in jobs])
    return ("\n\n" + "=" * 80 + "\n\n").join(results)


async def get_failed_job_urls_for_pr(
    session: aiohttp.ClientSession,
    repo: str,
    pr_number: int,
) -> list[str]:
    """Get all failed job URLs for a PR."""
    log(f"Fetching https://github.com/{repo}/pull/{pr_number}")
    pr_details = await get_pr_details(session, pr_number, repo)
    head_sha = pr_details["head"]["sha"]
    log(f"PR head SHA: {head_sha[:8]}")

    runs = get_workflow_runs(session, repo, head_sha)
    failed_runs = [r async for r in runs if r.get("conclusion") == "failure"]
    log(f"Found {len(failed_runs)} failed workflow run(s)")

    failed_jobs_tasks = [get_failed_jobs(session, run["id"], repo) for run in failed_runs]
    failed_jobs_results = await asyncio.gather(*failed_jobs_tasks)

    run_job_pairs = [
        (run, job) for run, jobs in zip(failed_runs, failed_jobs_results) for job in jobs
    ]
    log(f"Found {len(run_job_pairs)} failed job(s)")

    return [
        f"https://github.com/{repo}/actions/runs/{run['id']}/job/{job['id']}"
        for run, job in run_job_pairs
    ]


async def cmd_analyze_async(
    pr_infos: list[tuple[str, int]] | None,
    job_urls: list[str] | None,
    github_token: str,
) -> None:
    async with aiohttp.ClientSession(headers=get_headers(github_token)) as session:
        # If PRs provided, list failed jobs for each
        if pr_infos:
            all_job_urls = []
            for repo, pr_number in pr_infos:
                urls = await get_failed_job_urls_for_pr(session, repo, pr_number)
                all_job_urls.extend(urls)
            job_urls = all_job_urls

        if not job_urls:
            log("No failed jobs found")
            return

        # Fetch logs for all jobs
        log(f"Fetching logs for {len(job_urls)} job(s)")
        results = await asyncio.gather(*[fetch_single_job_logs(session, url) for url in job_urls])

    # Analyze with Claude
    log("Analyzing logs with Claude...")
    summary = await analyze_with_claude(results)
    print(summary)


# ============================================================================
# Main
# ============================================================================


def validate_urls(urls: list[str]) -> tuple[str, list[tuple[str, int]] | list[str]]:
    """Validate URLs and return ("pr", [(repo, pr_number), ...]) or ("job", job_urls)."""
    first_url = urls[0]

    # Check if first URL is a PR URL
    if parse_pr_url(first_url):
        pr_infos = []
        for url in urls:
            if pr_info := parse_pr_url(url):
                pr_infos.append(pr_info)
            else:
                log(f"Error: Mixed URL types. Expected PR URL: {url}")
                sys.exit(1)
        return ("pr", pr_infos)

    # Validate all URLs are job URLs
    for url in urls:
        if not JOB_URL_PATTERN.search(url):
            log(f"Error: Invalid URL: {url}")
            log("Expected PR URL (github.com/owner/repo/pull/123)")
            log("Or job URL (github.com/owner/repo/actions/runs/123/job/456)")
            sys.exit(1)

    return ("job", urls)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and analyze logs from failed GitHub Action jobs"
    )
    parser.add_argument("urls", nargs="+", help="PR URL or job URL(s) to analyze")
    parser.add_argument("--github-token", help="GitHub token (or set GITHUB_TOKEN)")

    args = parser.parse_args()
    github_token = get_github_token(args.github_token)

    url_type, url_info = validate_urls(args.urls)

    if url_type == "pr":
        asyncio.run(
            cmd_analyze_async(
                pr_infos=url_info,
                job_urls=None,
                github_token=github_token,
            )
        )
    else:
        asyncio.run(
            cmd_analyze_async(
                pr_infos=None,
                job_urls=url_info,
                github_token=github_token,
            )
        )


if __name__ == "__main__":
    main()
