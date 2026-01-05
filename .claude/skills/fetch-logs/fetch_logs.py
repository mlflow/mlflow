"""
Fetch logs from failed GitHub Action jobs.

Usage:
    # List failed jobs for a PR (outputs JSON)
    uv run fetch_logs.py list <owner/repo> <pr_number>

    # Fetch logs for specific jobs (outputs JSON)
    uv run fetch_logs.py fetch <job_url> [job_url ...]
"""
# /// script
# dependencies = [
#     "aiohttp",
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
from dataclasses import asdict, dataclass
from typing import Any

import aiohttp

GITHUB_API_BASE = "https://api.github.com"


@dataclass
class FailedJob:
    workflow_name: str
    job_name: str
    job_url: str


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
    log(f"Compacted logs: {len(logs):,} chars")
    return logs


def get_failed_step(job_details: dict[str, Any]) -> dict[str, Any] | None:
    """Get the first failed step from job details."""
    steps = job_details.get("steps", [])
    for step in steps:
        if step.get("conclusion") == "failure":
            return step
    return None


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
# Subcommand: fetch-logs
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
    failed_step_name = failed_step.get("name")

    return JobLogs(
        workflow_name=workflow_name,
        job_name=job_name,
        job_url=job_url,
        failed_step=failed_step_name,
        logs=cleaned_logs,
    )


async def cmd_fetch_logs_async(job_urls: list[str], github_token: str) -> None:
    log(f"Fetching logs for {len(job_urls)} job(s)")

    async with aiohttp.ClientSession(headers=get_headers(github_token)) as session:
        results = await asyncio.gather(*[fetch_single_job_logs(session, url) for url in job_urls])

    output = {"jobs": [asdict(job) for job in results]}
    print(json.dumps(output, indent=2))


def cmd_fetch_logs(args: argparse.Namespace) -> None:
    github_token = get_github_token(args.github_token)
    asyncio.run(cmd_fetch_logs_async(args.job_urls, github_token))


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Fetch logs from failed GitHub Action jobs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List failed jobs (outputs JSON)")
    list_parser.add_argument("repo", help="Repository in owner/repo format")
    list_parser.add_argument("pr_number", type=int, help="Pull request number")
    list_parser.add_argument("--github-token", help="GitHub token (or set GITHUB_TOKEN)")
    list_parser.set_defaults(func=cmd_list)

    # fetch subcommand
    fetch_logs_parser = subparsers.add_parser(
        "fetch", help="Fetch logs for jobs by URL (outputs JSON)"
    )
    fetch_logs_parser.add_argument("job_urls", nargs="+", help="GitHub Actions job URLs")
    fetch_logs_parser.add_argument("--github-token", help="GitHub token (or set GITHUB_TOKEN)")
    fetch_logs_parser.set_defaults(func=cmd_fetch_logs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
