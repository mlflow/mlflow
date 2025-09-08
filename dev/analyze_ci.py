#!/usr/bin/env python
"""
Analyze MLflow CI workflow runs for performance optimization.

This script analyzes GitHub Actions workflow runs to help maintainers identify
performance bottlenecks, uneven load balancing, and optimization opportunities
in MLflow's continuous integration pipeline.

Features:
- Analyzes job durations across matrix splits
- Provides step-level timing analysis
- Calculates statistical metrics (min, max, average, variance)
- Groups matrix jobs for aggregate analysis
- Supports both human-readable and JSON output formats

Usage:
    # Analyze latest run from current branch
    uv run python dev/analyze_ci.py

    # Analyze specific run
    uv run python dev/analyze_ci.py --run-id 123456789

    # JSON output for scripting
    uv run python dev/analyze_ci.py --json
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime


def run_command(cmd: str, capture: bool = True, check: bool = True) -> str | None:
    """Run shell command and return output."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture, text=True, check=check)
        if capture:
            return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if check:
            print(f"Command failed: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")
            raise
        return None


def get_current_branch() -> str | None:
    """Get current Git branch name."""
    try:
        return run_command("git branch --show-current", check=False)
    except Exception:
        return None


def get_recent_run_id(repo: str, branch: str = "master", status: str = "completed") -> str | None:
    """Get the most recent run ID for a branch."""
    cmd = (
        f"gh run list --repo {repo} --branch {branch} --status {status} "
        f'--limit 1 --json databaseId --jq ".[0].databaseId"'
    )
    run_id = run_command(cmd, check=False)
    if run_id and run_id.isdigit():
        return run_id
    return None


def determine_run_id(repo: str, specified_run_id: str | None = None) -> str | None:
    """Determine which run ID to analyze."""
    if specified_run_id:
        print(f"Using specified run ID: {specified_run_id}")
        return specified_run_id

    # Try current branch first
    current_branch = get_current_branch()
    if current_branch and current_branch != "master":
        print(f"Finding latest run for current branch: {current_branch}")
        run_id = get_recent_run_id(repo, current_branch)
        if run_id:
            print(f"Found run ID on current branch: {run_id}")
            return run_id
        print(f"No successful runs found on branch {current_branch}, falling back to master")

    # Fall back to master
    print("Finding latest run on master branch")
    run_id = get_recent_run_id(repo, "master")
    if run_id:
        print(f"Found run ID on master: {run_id}")
        return run_id

    print("Error: No successful runs found")
    return None


def get_run_info(run_id: str, repo: str) -> dict[str, str]:
    """Get basic run information."""
    cmd = (
        f'gh api repos/{repo}/actions/runs/{run_id} '
        f'--jq "{{id, html_url, status, conclusion, created_at, updated_at}}"'
    )
    result = run_command(cmd)
    if result:
        return json.loads(result)
    return {}


def get_jobs_data(run_id: str, repo: str) -> list[dict]:
    """Get detailed job information from GitHub API."""
    cmd = f'gh api repos/{repo}/actions/runs/{run_id}/jobs --paginate --jq ".jobs[]"'
    result = run_command(cmd)
    if result:
        # Parse multiple JSON objects (one per line)
        jobs = []
        for line in result.strip().split("\n"):
            if line:
                jobs.append(json.loads(line))
        return jobs
    return []


def parse_job_name(job_name: str) -> tuple[str, str | None]:
    """Parse job name to extract base name and matrix identifier.

    Examples:
        'python (1)' -> ('python', '1')
        'genai (3)' -> ('genai', '3')
        'flavors' -> ('flavors', None)
    """
    # Match pattern: jobname (N)
    match = re.match(r"^(.+?)\s*\((\d+)\)$", job_name)
    if match:
        return match.group(1), match.group(2)
    return job_name, None


def calculate_duration_seconds(started_at: str, completed_at: str) -> int:
    """Calculate duration in seconds between two ISO timestamps."""

    start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
    return int((end - start).total_seconds())


def format_duration(seconds: int) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def calculate_stats(values: list[float]) -> dict[str, float]:
    """Calculate min, max, average, variance for a list of values."""
    if not values:
        return {"min": 0, "max": 0, "average": 0, "variance": 0}

    return {
        "min": min(values),
        "max": max(values),
        "average": statistics.mean(values),
        "variance": statistics.variance(values) if len(values) > 1 else 0,
    }


def analyze_jobs(jobs_data: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Analyze job data and group by matrix."""
    job_groups = {}

    for job in jobs_data:
        if job["status"] != "completed" or not job.get("started_at") or not job.get("completed_at"):
            continue

        job_name = job["name"]
        base_name, matrix_id = parse_job_name(job_name)
        duration = calculate_duration_seconds(job["started_at"], job["completed_at"])

        # Initialize job group if needed
        if base_name not in job_groups:
            job_groups[base_name] = {"durations": [], "matrix_ids": [], "steps": {}}

        # Add job duration
        job_groups[base_name]["durations"].append(duration)
        job_groups[base_name]["matrix_ids"].append(matrix_id or "single")

        # Process steps
        for step in job.get("steps", []):
            if (
            step["status"] != "completed"
            or not step.get("started_at")
            or not step.get("completed_at")
        ):
                continue

            step_name = step["name"]
            step_duration = calculate_duration_seconds(step["started_at"], step["completed_at"])

            if step_name not in job_groups[base_name]["steps"]:
                job_groups[base_name]["steps"][step_name] = []
            job_groups[base_name]["steps"][step_name].append(step_duration)

    # Calculate statistics for each job group
    analysis = {}
    for base_name, data in job_groups.items():
        analysis[base_name] = {
            "splits": len(data["durations"]),
            "durations": data["durations"],
            "matrix_ids": data["matrix_ids"],
            "statistics": calculate_stats(data["durations"]),
            "steps": {},
        }

        # Calculate step statistics
        for step_name, step_durations in data["steps"].items():
            analysis[base_name]["steps"][step_name] = calculate_stats(step_durations)

    return analysis


def print_human_readable(run_info: dict[str, str], analysis: dict[str, dict[str, str]]):
    """Print analysis in human-readable table format."""
    print(f"\nCI Run Analysis: {run_info.get('html_url', 'Unknown URL')}")

    # Calculate total duration
    all_durations = []
    for job_data in analysis.values():
        all_durations.extend(job_data["durations"])

    if all_durations:
        total_duration = max(all_durations)  # Approximate - jobs run in parallel
        print(f"Longest Job Duration: {format_duration(total_duration)}")

    # Job Summary Table
    print(f"\n{'JOB SUMMARY'}")
    print("┌─────────────────┬─────────┬─────────┬─────────┬─────────┬──────────┐")
    print("│ Job             │ Splits  │ Min     │ Max     │ Average │ Variance │")
    print("├─────────────────┼─────────┼─────────┼─────────┼─────────┼──────────┤")

    for job_name, data in sorted(analysis.items()):
        stats = data["statistics"]
        min_dur = format_duration(int(stats["min"]))
        max_dur = format_duration(int(stats["max"]))
        avg_dur = format_duration(int(stats["average"]))
        print(
            f"│ {job_name:<15} │ {data['splits']:<7} │ {min_dur:<7} "
            f"│ {max_dur:<7} │ {avg_dur:<7} │ {stats['variance']:<8.1f} │"
        )

    print("└─────────────────┴─────────┴─────────┴─────────┴─────────┴──────────┘")

    # Step Analysis for each job
    for job_name, data in sorted(analysis.items()):
        if not data["steps"] or data["splits"] < 2:
            continue

        print(f"\nSTEP ANALYSIS - {job_name}")
        print("┌──────────────────────────┬─────────┬─────────┬─────────┬──────────┐")
        print("│ Step                     │ Min     │ Max     │ Average │ Variance │")
        print("├──────────────────────────┼─────────┼─────────┼─────────┼──────────┤")

        for step_name, stats in sorted(data["steps"].items()):
            if stats["min"] > 0:  # Only show steps that actually ran
                step_display = step_name[:24]  # Truncate long names
                min_dur = format_duration(int(stats["min"]))
                max_dur = format_duration(int(stats["max"]))
                avg_dur = format_duration(int(stats["average"]))
                print(
                    f"│ {step_display:<24} │ {min_dur:<7} │ {max_dur:<7} "
                    f"│ {avg_dur:<7} │ {stats['variance']:<8.1f} │"
                )

        print("└──────────────────────────┴─────────┴─────────┴─────────┴──────────┘")


def print_json_output(run_info: dict[str, str], analysis: dict[str, dict[str, str]]):
    """Print analysis in JSON format."""
    output = {
        "run_id": run_info.get("id"),
        "run_url": run_info.get("html_url"),
        "status": run_info.get("status"),
        "conclusion": run_info.get("conclusion"),
        "jobs": analysis,
    }

    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MLflow CI workflow runs for performance optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze latest run from current branch (fallback to master)
  uv run python dev/analyze_ci.py

  # Analyze specific run
  uv run python dev/analyze_ci.py --run-id 123456789

  # JSON output for scripting
  uv run python dev/analyze_ci.py --json
""",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="GitHub Actions run ID to analyze",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )

    args = parser.parse_args()
    repo = "mlflow/mlflow"

    # Determine run ID to analyze
    run_id = determine_run_id(repo, args.run_id)
    if not run_id:
        return 1

    print(f"Analyzing run: https://github.com/{repo}/actions/runs/{run_id}")

    # Get run info and jobs data
    run_info = get_run_info(run_id, repo)
    jobs_data = get_jobs_data(run_id, repo)

    if not jobs_data:
        print("Error: No job data found for this run")
        return 1

    # Analyze the jobs
    analysis = analyze_jobs(jobs_data)

    if not analysis:
        print("Error: No completed jobs found to analyze")
        return 1

    # Output results
    if args.json:
        print_json_output(run_info, analysis)
    else:
        print_human_readable(run_info, analysis)

    return 0


if __name__ == "__main__":
    sys.exit(main())

