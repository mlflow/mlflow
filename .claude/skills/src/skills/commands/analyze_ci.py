# ruff: noqa: T201
"""Analyze failed GitHub Action jobs."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skills.github import GitHubClient, Job, JobStep, get_github_token

MAX_LOG_TOKENS = 100_000
CHARS_PER_TOKEN = 2
LOG_CACHE_DIR = Path(tempfile.gettempdir()) / "analyze-ci"
LOG_CACHE_TTL_SECONDS = 3 * 86400


@dataclass
class JobLogs:
    workflow_name: str
    job_name: str
    job_url: str
    failed_step: str | None
    logs: str
    raw_log_path: Path
    package_versions_path: Path | None
    conclusion: str | None


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

    logs = "\n".join(result)
    log(f"Compacted logs: {len(logs) // CHARS_PER_TOKEN:,} tokens")
    return logs


def truncate_logs(logs: str, max_tokens: int = MAX_LOG_TOKENS) -> str:
    """Truncate logs to fit within token limit, keeping the end (where errors are)."""
    estimated_tokens = len(logs) // CHARS_PER_TOKEN
    if estimated_tokens <= max_tokens:
        return logs
    log(f"Truncating logs from {estimated_tokens:,} to {max_tokens:,} tokens")
    truncated = logs[-(max_tokens * CHARS_PER_TOKEN) :]
    return f"(showing last {max_tokens:,} tokens)\n{truncated}"


PR_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/pull/(\d+)")
JOB_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/actions/runs/(\d+)/job/(\d+)")
RUN_URL_PATTERN = re.compile(r"github\.com/([^/]+/[^/]+)/actions/runs/(\d+)")


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
            run_jobs = [
                j async for j in client.get_jobs(owner, repo, run_id) if j.conclusion == "failure"
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
    package_versions_path = extract_package_versions(raw_log_path)

    failed_step = next((s for s in job.steps if s.conclusion == "failure"), None)
    if not failed_step:
        return JobLogs(
            workflow_name=job.workflow_name,
            job_name=job.name,
            job_url=job.html_url,
            failed_step=None,
            logs="",
            raw_log_path=raw_log_path,
            package_versions_path=package_versions_path,
            conclusion=job.conclusion,
        )
    cleaned_logs = compact_logs(iter_step_lines(raw_log_path, failed_step))
    truncated_logs = truncate_logs(cleaned_logs)

    return JobLogs(
        workflow_name=job.workflow_name,
        job_name=job.name,
        job_url=job.html_url,
        failed_step=failed_step.name,
        logs=truncated_logs,
        raw_log_path=raw_log_path,
        package_versions_path=package_versions_path,
        conclusion=job.conclusion,
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
    if job.failed_step is None:
        text = (
            f"## {job.workflow_name} / {job.job_name}\n"
            f"URL: {job.job_url}\n"
            f"Conclusion: {job.conclusion}\n\n"
            f"Job has no failure to analyze. Raw log cached at {job.raw_log_path}"
        )
        if job.package_versions_path:
            text = f"{text}\nPackage versions: {job.package_versions_path}"
        return AnalysisResult(text=text, total_cost_usd=None, usage=None)

    formatted_logs = format_single_job_for_analysis(job)
    prompt = f"Analyze this CI failure:\n\n{formatted_logs}"

    # Use an isolated temp directory to avoid conflicts with the parent Claude session
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = await asyncio.create_subprocess_exec(
            "claude",
            "--print",
            "--model",
            "haiku",
            "--system-prompt",
            ANALYZE_SYSTEM_PROMPT,
            "--tools",
            "",
            "--output-format",
            "json",
            cwd=tmpdir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(prompt.encode("utf-8"))
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude exited with code {proc.returncode}: "
                f"{stderr.decode('utf-8', errors='replace')}"
            )

    data = json.loads(stdout)
    text = data.get("result", "")
    # Surface the raw log path so downstream agents can grep it for deeper analysis
    text = f"{text}\n\nRaw log: {job.raw_log_path}"
    if job.package_versions_path:
        text = f"{text}\nPackage versions: {job.package_versions_path}"
    return AnalysisResult(
        text=text,
        total_cost_usd=data.get("total_cost_usd"),
        usage=data.get("usage"),
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
    prune_old_cached_logs()
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
    parser.add_argument("urls", nargs="+", help="PR URL, workflow run URL, or job URL(s)")
    parser.add_argument("--debug", action="store_true", help="Show token/cost info")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    asyncio.run(cmd_analyze_async(args.urls, args.debug))
