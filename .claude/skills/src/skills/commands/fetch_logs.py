# ruff: noqa: T201
"""Fetch logs from failed GitHub Action jobs for downstream analysis."""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from skills.github import GitHubClient, Job, JobStep, get_github_token

LOG_CACHE_DIR = Path(tempfile.gettempdir()) / "fetch-logs"
LOG_CACHE_TTL_SECONDS = 3 * 86400


@dataclass
class JobLogs:
    workflow_name: str
    job_name: str
    job_url: str
    has_failed_steps: bool
    raw_log_path: Path
    failed_step_lines: list[tuple[int, int]]
    pytest_summary_lines: list[tuple[int, int]]
    package_versions_lines: list[tuple[int, int]]
    snapshot_changes_lines: list[tuple[int, int]]
    conclusion: str | None


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ?")
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;:]*m")
PYTEST_SECTION_PATTERN = re.compile(r"^=+ (.+?) =+$")


def strip_log_colors(log_path: Path) -> None:
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    cleaned = ANSI_PATTERN.sub("", raw)
    if cleaned != raw:
        log_path.write_text(cleaned, encoding="utf-8")


def to_seconds(ts: str) -> str:
    """Truncate timestamp to seconds precision for comparison."""
    return ts[:19]  # "2026-01-05T07:17:56.1234567Z" -> "2026-01-05T07:17:56"


def prune_old_cached_logs() -> None:
    """Delete cached raw logs older than LOG_CACHE_TTL_SECONDS and empty run dirs."""
    if not LOG_CACHE_DIR.exists():
        return
    cutoff = time.time() - LOG_CACHE_TTL_SECONDS
    for cached_file in LOG_CACHE_DIR.rglob("*"):
        if cached_file.is_file() and cached_file.stat().st_mtime < cutoff:
            cached_file.unlink()
    for run_dir in LOG_CACHE_DIR.iterdir():
        if run_dir.is_dir() and not any(run_dir.iterdir()):
            run_dir.rmdir()


async def download_raw_log(client: GitHubClient, job: Job) -> Path:
    """Download the full raw log to the cache dir, or return cached path if present."""
    log_path = LOG_CACHE_DIR / str(job.run_id) / f"{job.id}.log"
    if log_path.exists():
        strip_log_colors(log_path)
        log(f"Using cached raw log at {log_path}")
        return log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename on success so an interrupted download
    # doesn't leave a partial file that future runs would silently reuse.
    tmp_path = log_path.with_suffix(".log.tmp")
    async with await client.get_raw(f"{job.url}/logs") as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            async for chunk in response.content.iter_chunked(64 * 1024):
                f.write(chunk)
    strip_log_colors(tmp_path)
    tmp_path.rename(log_path)
    log(f"Saved raw log to {log_path}")
    return log_path


def step_line_range(log_path: Path, failed_step: JobStep) -> tuple[int, int] | None:
    """Find the inclusive, 1-based range for a failed step in the saved log."""
    if not failed_step.started_at or not failed_step.completed_at:
        return None

    # ISO 8601 timestamps are lexicographically sortable, so we can compare as strings
    start_secs = to_seconds(failed_step.started_at)
    end_secs = to_seconds(failed_step.completed_at)
    in_range = False
    start_line = None
    end_line = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, 1):
            if TIMESTAMP_PATTERN.match(line):
                ts_secs = to_seconds(line)
                if ts_secs > end_secs:
                    break
                in_range = ts_secs >= start_secs
            if in_range:
                if start_line is None:
                    start_line = line_number
                end_line = line_number
    return (start_line, end_line) if start_line is not None and end_line is not None else None


PACKAGE_VERSIONS_BEGIN_MARKER = ">>> package versions"
PACKAGE_VERSIONS_END_MARKER = "<<< package versions"
SNAPSHOT_CHANGES_BEGIN_MARKER = ">>> snapshot changes"
SNAPSHOT_CHANGES_END_MARKER = "<<< snapshot changes"


def log_block_ranges(log_path: Path, begin_marker: str, end_marker: str) -> list[tuple[int, int]]:
    ranges = []
    start_line = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, 1):
            content = TIMESTAMP_PATTERN.sub("", line.rstrip("\r\n"))
            if content == begin_marker:
                start_line = line_number + 1
            elif content == end_marker and start_line is not None:
                if start_line < line_number:
                    ranges.append((start_line, line_number - 1))
                start_line = None
    return ranges


def pytest_summary_ranges(log_path: Path) -> list[tuple[int, int]]:
    ranges = []
    start_line = None
    line_number = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, 1):
            content = TIMESTAMP_PATTERN.sub("", line.rstrip("\r\n")).strip()
            if match := PYTEST_SECTION_PATTERN.match(content):
                section = match.group(1).strip()
                is_result = re.search(r"\b(?:\d+ .+|no tests ran) in ", section) is not None
                if start_line is not None:
                    # Include the final result banner, but not the next section's header.
                    end_line = line_number if is_result else line_number - 1
                    ranges.append((start_line, end_line))
                    start_line = None
                elif is_result:
                    ranges.append((line_number, line_number))
                if section == "short test summary info":
                    start_line = line_number
    if start_line is not None:
        ranges.append((start_line, line_number))
    return ranges


PR_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/pull/(\d+)")
JOB_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/actions/runs/(\d+)/job/(\d+)(?=[?#]|$)")
# Run URLs may optionally specify an attempt (`/attempts/{n}`); when omitted, GitHub's
# default behavior is to return jobs from the latest attempt only.
RUN_URL_PATTERN = re.compile(
    r"github\.com/([^/]+/[^/]+)/actions/runs/(\d+)(?:/attempts/(\d+))?(?=[?#]|$)"
)


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
        elif match := RUN_URL_PATTERN.search(url):
            repo_full = match.group(1)
            owner, repo = repo_full.split("/")
            run_id = int(match.group(2))
            attempt = int(match.group(3)) if match.group(3) else None
            run_jobs = [
                j
                async for j in client.get_jobs(owner, repo, run_id, attempt)
                if j.conclusion == "failure"
            ]
            jobs.extend(run_jobs)
        elif match := PR_URL_PATTERN.search(url):
            repo_full = match.group(1)
            owner, repo = repo_full.split("/")
            pr_jobs = await get_failed_jobs_from_pr(client, owner, repo, int(match.group(2)))
            jobs.extend(pr_jobs)
        else:
            log(f"Error: Invalid URL: {url}")
            log("Expected PR URL (github.com/owner/repo/pull/123)")
            log("Or workflow run URL (github.com/owner/repo/actions/runs/123)")
            log("Or job URL (github.com/owner/repo/actions/runs/123/job/456)")
            sys.exit(1)

    return jobs


async def fetch_single_job_logs(client: GitHubClient, job: Job) -> JobLogs:
    log(f"Fetching logs for '{job.workflow_name} / {job.name}'")
    raw_log_path = await download_raw_log(client, job)
    failed_steps = [s for s in job.steps if s.conclusion == "failure"]

    return JobLogs(
        workflow_name=job.workflow_name,
        job_name=job.name,
        job_url=job.html_url,
        has_failed_steps=bool(failed_steps),
        raw_log_path=raw_log_path,
        failed_step_lines=[
            line_range
            for step in failed_steps
            if (line_range := step_line_range(raw_log_path, step)) is not None
        ],
        pytest_summary_lines=pytest_summary_ranges(raw_log_path),
        package_versions_lines=log_block_ranges(
            raw_log_path, PACKAGE_VERSIONS_BEGIN_MARKER, PACKAGE_VERSIONS_END_MARKER
        ),
        snapshot_changes_lines=log_block_ranges(
            raw_log_path, SNAPSHOT_CHANGES_BEGIN_MARKER, SNAPSHOT_CHANGES_END_MARKER
        ),
        conclusion=job.conclusion,
    )


def format_job_output(job: JobLogs) -> str:
    parts = [
        f"## {job.workflow_name} / {job.job_name}",
        f"URL: {job.job_url}",
    ]
    if not job.has_failed_steps:
        parts.append(f"Conclusion: {job.conclusion} (no failed step recorded)")
    parts.append(f"Raw log: {job.raw_log_path}")
    for label, ranges in [
        ("Failed step lines", job.failed_step_lines),
        ("Pytest summary lines", job.pytest_summary_lines),
        ("Package versions lines", job.package_versions_lines),
        ("Snapshot changes lines", job.snapshot_changes_lines),
    ]:
        if ranges:
            parts.append(f"{label}: {', '.join(f'{start}-{end}' for start, end in ranges)}")
    return "\n".join(parts)


async def cmd_fetch_async(urls: list[str]) -> None:
    prune_old_cached_logs()
    github_token = get_github_token()
    async with GitHubClient(github_token) as client:
        jobs = await resolve_urls(client, urls)

        if not jobs:
            log("No failed jobs found")
            return

        log(f"Fetching logs for {len(jobs)} job(s)")
        results = await asyncio.gather(*[fetch_single_job_logs(client, job) for job in jobs])

    separator = "\n\n---\n\n"
    print(separator.join(format_job_output(r) for r in results))


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("fetch-logs", help="Fetch logs from failed CI jobs")
    parser.add_argument("urls", nargs="+", help="PR URL, workflow run URL, or job URL(s)")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(cmd_fetch_async(args.urls))
