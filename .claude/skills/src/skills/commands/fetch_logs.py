# ruff: noqa: T201
"""Fetch logs from failed GitHub Action jobs for downstream analysis."""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator
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
    failed_step: str | None
    raw_log_path: Path
    failed_step_log_path: Path | None
    package_versions_path: Path | None
    conclusion: str | None


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z ?")


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
    tmp_path.rename(log_path)
    log(f"Saved raw log to {log_path}")
    return log_path


def iter_step_lines(log_path: Path, failed_step: JobStep) -> Iterator[str]:
    """Yield lines from the saved log file filtered to the failed step's time range."""
    if not failed_step.started_at or not failed_step.completed_at:
        raise ValueError("Failed step missing timestamps")

    # ISO 8601 timestamps are lexicographically sortable, so we can compare as strings
    start_secs = to_seconds(failed_step.started_at)
    end_secs = to_seconds(failed_step.completed_at)
    in_range = False

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if TIMESTAMP_PATTERN.match(line):
                ts_secs = to_seconds(line)
                if ts_secs > end_secs:
                    return  # Past end time, stop reading
                in_range = ts_secs >= start_secs
                # Use partition so a bare-timestamp line (no trailing content) yields ""
                _, _, line = line.partition(" ")
            if in_range:
                yield line


PACKAGE_VERSIONS_BEGIN_MARKER = ">>> package versions"
PACKAGE_VERSIONS_END_MARKER = "<<< package versions"


def extract_package_versions(log_path: Path) -> Path | None:
    """Save the show-versions action's package list block next to the raw log."""
    out_path = log_path.with_suffix(".package-versions.txt")
    if out_path.exists():
        log(f"Using cached package versions at {out_path}")
        return out_path

    captured: list[str] = []
    capturing = False
    terminated = False
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            content = TIMESTAMP_PATTERN.sub("", line)
            if not capturing:
                if content == PACKAGE_VERSIONS_BEGIN_MARKER:
                    capturing = True
                continue
            if content == PACKAGE_VERSIONS_END_MARKER:
                terminated = True
                break
            captured.append(content)

    if not terminated or not captured:
        return None
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(captured) + "\n")
    log(f"Saved package versions to {out_path}")
    return out_path


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
PYTEST_SECTION_PATTERN = re.compile(r"^={6,} (.+?) ={6,}$")

# Pytest sections to skip (not useful for failure analysis)
PYTEST_SKIP_SECTIONS = {
    "test session starts",
    "warnings summary",
    "per-file durations",
    "remaining threads",
}


def compact_logs(lines: Iterable[str]) -> str:
    """Clean logs: strip timestamps, ANSI colors, and filter noisy pytest sections."""
    result: list[str] = []
    skip_section = False

    for line in lines:
        line = ANSI_PATTERN.sub("", line)
        if match := PYTEST_SECTION_PATTERN.match(line.strip()):
            section_name = match.group(1).strip().lower()
            skip_section = any(name in section_name for name in PYTEST_SKIP_SECTIONS)
        if not skip_section:
            result.append(line)

    return "\n".join(result)


PR_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/pull/(\d+)")
# Job URLs may optionally embed an attempt segment (`/attempts/{n}`) before `/job/{id}`.
# The job ID is unique across attempts so the attempt number isn't needed for lookup.
JOB_URL_PATTERN = re.compile(
    r"github\.com/([^/]+/[^/]+)/actions/runs/(\d+)(?:/attempts/\d+)?/job/(\d+)"
)
# Run URLs may optionally specify an attempt (`/attempts/{n}`); when omitted, GitHub's
# default behavior is to return jobs from the latest attempt only.
RUN_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/actions/runs/(\d+)(?:/attempts/(\d+))?")


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


def extract_failed_step_log(raw_log_path: Path, failed_step: JobStep) -> Path:
    out_path = raw_log_path.with_suffix(".failed-step.log")
    if out_path.exists():
        log(f"Using cached failed-step log at {out_path}")
        return out_path
    cleaned = compact_logs(iter_step_lines(raw_log_path, failed_step))
    out_path.write_text(cleaned, encoding="utf-8")
    log(f"Saved failed-step log to {out_path}")
    return out_path


async def fetch_single_job_logs(client: GitHubClient, job: Job) -> JobLogs:
    log(f"Fetching logs for '{job.workflow_name} / {job.name}'")
    raw_log_path = await download_raw_log(client, job)
    package_versions_path = extract_package_versions(raw_log_path)

    failed_step = next((s for s in job.steps if s.conclusion == "failure"), None)
    failed_step_log_path = (
        extract_failed_step_log(raw_log_path, failed_step) if failed_step else None
    )

    return JobLogs(
        workflow_name=job.workflow_name,
        job_name=job.name,
        job_url=job.html_url,
        failed_step=failed_step.name if failed_step else None,
        raw_log_path=raw_log_path,
        failed_step_log_path=failed_step_log_path,
        package_versions_path=package_versions_path,
        conclusion=job.conclusion,
    )


def format_job_output(job: JobLogs) -> str:
    parts = [
        f"## {job.workflow_name} / {job.job_name}",
        f"URL: {job.job_url}",
    ]
    if job.failed_step:
        parts.append(f"Failed step: {job.failed_step}")
    else:
        parts.append(f"Conclusion: {job.conclusion} (no failed step recorded)")
    parts.append(f"Raw log: {job.raw_log_path}")
    if job.failed_step_log_path:
        parts.append(f"Failed step log: {job.failed_step_log_path}")
    if job.package_versions_path:
        parts.append(f"Package versions: {job.package_versions_path}")
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
