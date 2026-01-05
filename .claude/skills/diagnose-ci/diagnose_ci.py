"""
Diagnose GitHub Action run failures.

Usage:
    # List failed jobs for a PR (outputs JSON)
    uv run diagnose_ci.py list <owner/repo> <pr_number>

    # Summarize a specific job by URL (outputs Markdown)
    OPENAI_API_KEY=... uv run diagnose_ci.py summarize <job_url>
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "aiohttp",
#     "openai",
#     "tiktoken",
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
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import aiohttp
import tiktoken
from openai import AsyncOpenAI

GITHUB_API_BASE = "https://api.github.com"


@dataclass
class FailedJob:
    workflow_name: str
    job_name: str
    job_url: str


@dataclass
class JobSummary:
    workflow_name: str
    job_name: str
    job_url: str
    failed_step: str | None
    input_tokens: int
    output_tokens: int
    cost: float
    truncated: bool
    summary: str
    logs: str | None = None


@dataclass
class SummaryResult:
    summary: str
    input_tokens: int
    output_tokens: int
    cost: float
    truncated: bool


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


def get_openai_api_key(cli_key: str | None = None) -> str:
    key = cli_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        log("Error: OPENAI_API_KEY is required (--openai-api-key or env var)")
        sys.exit(1)
    return key


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
) -> list[dict[str, Any]]:
    all_runs: list[dict[str, Any]] = []
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

        all_runs.extend(runs)
        page += 1

        if len(runs) < per_page:
            break

    return all_runs


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


async def get_job_logs(session: aiohttp.ClientSession, job_id: int, repo: str) -> str:
    url = f"{GITHUB_API_BASE}/repos/{repo}/actions/jobs/{job_id}/logs"
    async with session.get(url, allow_redirects=True) as response:
        response.raise_for_status()
        return await response.text()


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP_STRIP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ?", re.MULTILINE)
PYTEST_SECTION_PATTERN = re.compile(r"^={6,} (.+?) ={6,}$")


def strip_ansi(logs: str) -> str:
    return ANSI_PATTERN.sub("", logs)


def strip_timestamps(logs: str) -> str:
    return TIMESTAMP_STRIP_PATTERN.sub("", logs)


# Pytest sections to skip (not useful for failure analysis)
PYTEST_SKIP_SECTIONS = {
    "test session starts",
    "warnings summary",
    "per-file durations",
    "remaining threads",
}


def filter_pytest_sections(logs: str) -> str:
    """Remove noisy pytest sections like 'test session starts'."""
    lines = logs.split("\n")
    result: list[str] = []
    skip = False

    for line in lines:
        if match := PYTEST_SECTION_PATTERN.match(line.strip()):
            section_name = match.group(1).strip().lower()
            skip = any(name in section_name for name in PYTEST_SKIP_SECTIONS)
        if not skip:
            result.append(line)

    return "\n".join(result)


def compact_logs(
    raw_logs: str,
    *,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> str:
    """Clean logs and remove noise."""
    # 1. Strip ANSI colors
    logs = strip_ansi(raw_logs)
    original_tokens = count_tokens(strip_timestamps(logs))

    # 2. Filter by time range (before stripping timestamps)
    if started_at and completed_at:
        logs = filter_by_time_range(logs, started_at, completed_at)

    # 3. Strip timestamps
    logs = strip_timestamps(logs)

    # 4. Filter pytest sections
    logs = filter_pytest_sections(logs)

    final_tokens = count_tokens(logs)
    log(f"Compacted logs: {original_tokens:,} -> {final_tokens:,} tokens")

    return logs


def get_failed_step(job_details: dict[str, Any]) -> dict[str, Any] | None:
    """Get the first failed step from job details."""
    steps = job_details.get("steps", [])
    for step in steps:
        if step.get("conclusion") == "failure":
            return step
    return None


TIMESTAMP_EXTRACT_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)")
MICROSECOND_TRUNCATE_PATTERN = re.compile(r"(\.\d{6})\d+Z$")


def parse_timestamp(ts: str) -> datetime:
    # Truncate to 6 digits for microseconds (Python %f limit)
    ts = MICROSECOND_TRUNCATE_PATTERN.sub(r"\1Z", ts)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts}")


def filter_by_time_range(logs: str, started_at: str, completed_at: str) -> str:
    """Filter logs to only include lines within the given time range."""
    start_time = parse_timestamp(started_at)
    end_time = parse_timestamp(completed_at)
    filtered_lines: list[str] = []
    in_range = False

    for line in logs.split("\n"):
        if match := TIMESTAMP_EXTRACT_PATTERN.match(line):
            try:
                line_time = parse_timestamp(match.group(1))
                in_range = start_time <= line_time <= end_time
            except ValueError:
                pass  # Keep previous in_range state
        if in_range:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


JOB_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/actions/runs/(\d+)/job/(\d+)")


def parse_job_url(url: str) -> tuple[str, int, int]:
    """Parse a GitHub Actions job URL and return (repo, run_id, job_id)."""
    if not (match := JOB_URL_PATTERN.search(url)):
        raise ValueError(f"Invalid job URL: {url}")
    return match.group(1), int(match.group(2)), int(match.group(3))


def build_failed_job(repo: str, run: dict[str, Any], job: dict[str, Any]) -> FailedJob:
    return FailedJob(
        workflow_name=run.get("name", "Unknown workflow"),
        job_name=job.get("name", "Unknown job"),
        job_url=f"https://github.com/{repo}/actions/runs/{run['id']}/job/{job['id']}",
    )


# Rough cost: ~$0.02/job (gpt-4.1-nano), ~$0.08/job (gpt-4.1-mini)
MAX_LOG_TOKENS = 200_000
tokenizer = tiktoken.encoding_for_model("gpt-4.1")

# Pricing per 1M tokens (input, cached, output)
# https://platform.openai.com/docs/pricing (as of 2025/01/05, may change)
MODEL_PRICING = {
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
}


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def truncate_logs(logs: str, max_tokens: int = MAX_LOG_TOKENS) -> tuple[str, bool]:
    """Truncate logs to fit within token limit, keeping the end (where errors are)."""
    tokens = tokenizer.encode(logs)
    if len(tokens) <= max_tokens:
        return logs, False

    return tokenizer.decode(tokens[-max_tokens:]), True


def compute_cost(
    model: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int
) -> float:
    """Compute cost in dollars based on model pricing."""
    input_price, cached_price, output_price = MODEL_PRICING[model]
    uncached_tokens = prompt_tokens - cached_tokens
    return (
        uncached_tokens * input_price
        + cached_tokens * cached_price
        + completion_tokens * output_price
    ) / 1_000_000


async def summarize_logs(
    client: AsyncOpenAI,
    logs: str,
    workflow_name: str,
    job_name: str,
    model: str = "gpt-4.1-mini",
) -> SummaryResult:
    logs, truncated = truncate_logs(logs)
    if truncated:
        log(f"Logs truncated to ~{MAX_LOG_TOKENS:,} tokens")

    prompt = f"""Analyze the following GitHub Actions failure logs and provide a concise summary.

Workflow: {workflow_name}
Job: {job_name}

Focus on:
1. The root cause of the failure
2. The specific error messages
3. Include relevant log snippets (e.g., assertion errors, stack traces)
4. For pytest failures, include full test names if available (e.g., tests/test_foo.py::test_bar)

Logs:
{logs}

Provide a clear summary in 1-2 short paragraphs in Markdown format."""

    log(f"Sending {count_tokens(prompt):,} tokens to OpenAI")
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    usage = response.usage
    cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
    cost = compute_cost(model, usage.prompt_tokens, cached_tokens, usage.completion_tokens)
    log(
        f"Response: {usage.prompt_tokens:,} in ({cached_tokens:,} cached), "
        f"{usage.completion_tokens:,} out, ${cost:.4f}"
    )
    return SummaryResult(
        summary=response.choices[0].message.content,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost=cost,
        truncated=truncated,
    )


# ============================================================================
# Subcommand: list
# ============================================================================


async def cmd_list_async(repo: str, pr_number: int, github_token: str) -> None:
    log(f"Fetching https://github.com/{repo}/pull/{pr_number}")

    async with aiohttp.ClientSession(headers=get_headers(github_token)) as session:
        pr_details = await get_pr_details(session, pr_number, repo)
        head_sha = pr_details["head"]["sha"]
        log(f"PR head SHA: {head_sha[:8]}")

        runs = await get_workflow_runs(session, repo, head_sha)
        failed_runs = [run for run in runs if run.get("conclusion") == "failure"]
        log(f"Found {len(failed_runs)} failed workflow run(s)")

        failed_jobs_tasks = [get_failed_jobs(session, run["id"], repo) for run in failed_runs]
        failed_jobs_results = await asyncio.gather(*failed_jobs_tasks)

        run_job_pairs = [
            (run, job) for run, jobs in zip(failed_runs, failed_jobs_results) for job in jobs
        ]
        log(f"Found {len(run_job_pairs)} failed job(s)")

        failed_jobs = [build_failed_job(repo, run, job) for run, job in run_job_pairs]

    output = {
        "pr": {
            "number": pr_number,
            "title": pr_details["title"],
            "branch": pr_details["head"]["ref"],
            "url": pr_details["html_url"],
        },
        "failed_jobs": [asdict(job) for job in failed_jobs],
    }
    print(json.dumps(output, indent=2))


def cmd_list(args: argparse.Namespace) -> None:
    github_token = get_github_token(args.github_token)
    asyncio.run(cmd_list_async(args.repo, args.pr_number, github_token))


# ============================================================================
# Subcommand: summarize
# ============================================================================


async def get_job_details(session: aiohttp.ClientSession, repo: str, job_id: int) -> dict[str, Any]:
    return await api_get(session, f"/repos/{repo}/actions/jobs/{job_id}")


async def get_run_details(session: aiohttp.ClientSession, repo: str, run_id: int) -> dict[str, Any]:
    return await api_get(session, f"/repos/{repo}/actions/runs/{run_id}")


async def summarize_single_job(
    session: aiohttp.ClientSession,
    client: AsyncOpenAI,
    job_url: str,
    model: str,
    verbose: bool = False,
) -> JobSummary:
    repo, run_id, job_id = parse_job_url(job_url)
    log(f"Fetching job {job_id} from {repo}")

    job_details = await get_job_details(session, repo, job_id)
    run_details = await get_run_details(session, repo, run_id)

    workflow_name = run_details.get("name", "Unknown workflow")
    job_name = job_details.get("name", "Unknown job")
    failed_step = get_failed_step(job_details)

    log(f"Fetching logs for '{workflow_name} / {job_name}'")
    raw_logs = await get_job_logs(session, job_id, repo)

    cleaned_logs = compact_logs(
        raw_logs,
        started_at=failed_step.get("started_at") if failed_step else None,
        completed_at=failed_step.get("completed_at") if failed_step else None,
    )
    failed_step_name = failed_step.get("name") if failed_step else None

    result = await summarize_logs(client, cleaned_logs, workflow_name, job_name, model)
    truncated_logs, _ = truncate_logs(cleaned_logs)
    return JobSummary(
        workflow_name=workflow_name,
        job_name=job_name,
        job_url=job_url,
        failed_step=failed_step_name,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost=result.cost,
        truncated=result.truncated,
        summary=result.summary,
        logs=truncated_logs if verbose else None,
    )


async def cmd_summarize_async(
    job_urls: list[str],
    model: str,
    github_token: str,
    openai_api_key: str,
    verbose: bool = False,
) -> None:
    log(f"Summarizing {len(job_urls)} job(s) with {model}")

    client = AsyncOpenAI(api_key=openai_api_key)

    async with aiohttp.ClientSession(headers=get_headers(github_token)) as session:
        results = await asyncio.gather(
            *[summarize_single_job(session, client, url, model, verbose) for url in job_urls]
        )

    for i, job in enumerate(results):
        if i > 0:
            print("\n---\n")
        print(f"# {job.workflow_name} - {job.job_name}")
        print()
        if job.failed_step:
            print(f"**Failed step:** {job.failed_step}")
        print(f"**URL:** {job.job_url}")
        truncated_note = " (truncated)" if job.truncated else ""
        print(f"**Tokens:** {job.input_tokens:,} in, {job.output_tokens:,} out{truncated_note}")
        print(f"**Cost:** ${job.cost:.4f}")
        print()
        print(job.summary)
        if job.logs:
            print()
            print("## Logs")
            print()
            print("```")
            print(job.logs)
            print("```")


def cmd_summarize(args: argparse.Namespace) -> None:
    github_token = get_github_token(args.github_token)
    openai_api_key = get_openai_api_key(args.openai_api_key)
    asyncio.run(
        cmd_summarize_async(args.job_urls, args.model, github_token, openai_api_key, args.verbose)
    )


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Summarize GitHub Action failures for a pull request"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List failed jobs (outputs JSON)")
    list_parser.add_argument("repo", help="Repository in owner/repo format")
    list_parser.add_argument("pr_number", type=int, help="Pull request number")
    list_parser.add_argument("--github-token", help="GitHub token (or set GITHUB_TOKEN)")
    list_parser.set_defaults(func=cmd_list)

    # summarize subcommand
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize jobs by URL (outputs Markdown)"
    )
    summarize_parser.add_argument("job_urls", nargs="+", help="GitHub Actions job URLs")
    summarize_parser.add_argument(
        "--model",
        "-m",
        default="gpt-4.1-mini",
        choices=["gpt-4.1-nano", "gpt-4.1-mini"],
        help="OpenAI model (default: gpt-4.1-mini)",
    )
    summarize_parser.add_argument("--github-token", help="GitHub token (or set GITHUB_TOKEN)")
    summarize_parser.add_argument("--openai-api-key", help="OpenAI API key (or set OPENAI_API_KEY)")
    summarize_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Include logs in the output"
    )
    summarize_parser.set_defaults(func=cmd_summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
