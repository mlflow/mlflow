#!/usr/bin/env python
"""
Analyze MLflow CI workflow runs for performance optimization.

This script analyzes GitHub Actions workflow runs to help maintainers identify
performance bottlenecks, uneven load balancing, and optimization opportunities
in MLflow's continuous integration pipeline.

Features:
- Analyzes job durations across matrix splits
- Provides step-level timing analysis for each job
- Downloads and analyzes Python test durations from CI artifacts
- Calculates statistical metrics (min, max, average, variance)
- Groups matrix jobs for aggregate analysis
- Identifies slowest tests and test distribution across groups
- Supports both human-readable and JSON output formats

Usage:
    # Analyze latest run from current branch
    uv run python dev/analyze_ci.py

    # Analyze specific run
    uv run python dev/analyze_ci.py --run-id 123456789

    # JSON output for scripting (includes all test data)
    uv run python dev/analyze_ci.py --json

    # Customize number of slowest tests shown (default: 20)
    uv run python dev/analyze_ci.py --top-n 50
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

# Map job names to their configuration for test duration artifacts
JOB_CONFIG = {
    "python": {"groups": 10},
    "python-skinny-tests": {"groups": 5},
    "flavors": {"groups": 10},
    "models": {"groups": 10},
    "evaluate": {"groups": 5},
    "genai": {"groups": 4},
    "pyfunc": {"groups": 10},
    "pyfunc-pydanticv1": {"groups": 4},
    "sagemaker": {"groups": 4},
    "windows": {"groups": 10},
}


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
        f"gh api repos/{repo}/actions/runs/{run_id} "
        f'--jq "{{id, html_url, status, conclusion, created_at, updated_at}}"'
    )
    result = run_command(cmd)
    if result:
        return json.loads(result)
    return {}


def get_jobs_data(run_id: str, repo: str) -> list[dict[str, Any]]:
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


def download_test_durations(run_id: str, repo: str) -> dict[str, dict[str, Any]]:
    """Download test duration artifacts from CI run and return parsed data.

    Returns a dict mapping job names to their test durations by group.
    Example: {"python": {1: {"test_name": duration, ...}, 2: {...}}}
    """
    print("\nDownloading test duration artifacts...")
    test_data = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for job_name, config in JOB_CONFIG.items():
            job_test_data = {}
            groups_found = 0

            for group in range(1, config["groups"] + 1):
                artifact_name = f"test-durations-{job_name}-group-{group}"

                # Try to download the artifact
                download_cmd = (
                    f"cd {tmpdir} && gh run download {run_id} --repo {repo} "
                    f"--name {artifact_name} 2>/dev/null"
                )
                result = run_command(download_cmd, check=False)

                if result is not None:
                    # The artifact is downloaded as group_N_durations.json
                    duration_file = Path(tmpdir) / f"group_{group}_durations.json"
                    if duration_file.exists():
                        groups_found += 1
                        with open(duration_file) as f:
                            group_durations = json.load(f)
                        if isinstance(group_durations, dict):
                            job_test_data[group] = group_durations
                        # Clean up to avoid conflicts with next download
                        duration_file.unlink()

            if groups_found > 0:
                test_data[job_name] = job_test_data
                print(f"  {job_name}: Found {groups_found}/{config['groups']} groups")

    return test_data


def analyze_test_durations(test_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Analyze test duration data to generate statistics and groupings.

    Returns comprehensive test analysis including by-file, by-group, and overall stats.
    """
    analysis = {
        "summary": {
            "total_tests": 0,
            "total_duration_seconds": 0.0,
            "average_duration": 0.0,
        },
        "by_file": {},
        "by_group": {},
        "all_tests": [],
    }

    # Process each job's test data
    for job_name, job_groups in test_data.items():
        for group_num, group_tests in job_groups.items():
            # Initialize group data if needed
            group_key = f"{job_name}-{group_num}"
            if group_key not in analysis["by_group"]:
                analysis["by_group"][group_key] = {
                    "job": job_name,
                    "group": group_num,
                    "test_count": 0,
                    "total_duration": 0.0,
                    "tests": [],
                }

            # Process each test
            for test_name, duration in group_tests.items():
                # Extract file name from test path
                file_name = test_name.split("::")[0] if "::" in test_name else test_name

                # Update summary stats
                analysis["summary"]["total_tests"] += 1
                analysis["summary"]["total_duration_seconds"] += duration

                # Create test entry
                test_entry = {
                    "name": test_name,
                    "duration": duration,
                    "file": file_name,
                    "job": job_name,
                    "group": group_num,
                }

                # Add to all_tests
                analysis["all_tests"].append(test_entry)

                # Update by_file stats
                if file_name not in analysis["by_file"]:
                    analysis["by_file"][file_name] = {
                        "test_count": 0,
                        "total_duration": 0.0,
                        "average_duration": 0.0,
                        "slowest_test": None,
                        "tests": [],
                    }

                analysis["by_file"][file_name]["test_count"] += 1
                analysis["by_file"][file_name]["total_duration"] += duration
                analysis["by_file"][file_name]["tests"].append(test_entry)

                # Update slowest test for file
                if (
                    analysis["by_file"][file_name]["slowest_test"] is None
                    or duration > analysis["by_file"][file_name]["slowest_test"]["duration"]
                ):
                    analysis["by_file"][file_name]["slowest_test"] = {
                        "name": test_name.split("::")[-1] if "::" in test_name else test_name,
                        "duration": duration,
                    }

                # Update by_group stats
                analysis["by_group"][group_key]["test_count"] += 1
                analysis["by_group"][group_key]["total_duration"] += duration
                analysis["by_group"][group_key]["tests"].append(test_name)

    # Calculate averages
    if analysis["summary"]["total_tests"] > 0:
        analysis["summary"]["average_duration"] = (
            analysis["summary"]["total_duration_seconds"] / analysis["summary"]["total_tests"]
        )

    for file_data in analysis["by_file"].values():
        if file_data["test_count"] > 0:
            file_data["average_duration"] = file_data["total_duration"] / file_data["test_count"]

    # Sort all_tests by duration (slowest first)
    analysis["all_tests"].sort(key=lambda x: x["duration"], reverse=True)

    # Sort tests within each file by duration
    for file_data in analysis["by_file"].values():
        file_data["tests"].sort(key=lambda x: x["duration"], reverse=True)

    return analysis


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


def print_test_analysis(test_analysis: dict[str, Any], top_n: int = 20):
    """Print test analysis in human-readable table format."""
    if not test_analysis or not test_analysis.get("all_tests"):
        print("\nNo test duration data available")
        return

    summary = test_analysis["summary"]
    print(f"\nPYTHON TEST SUMMARY")
    print(f"Total Tests: {summary['total_tests']:,}")
    print(f"Total Duration: {format_duration(int(summary['total_duration_seconds']))}")
    print(f"Average Duration: {summary['average_duration']:.2f}s")

    # Test file summary (top 10 files by total duration)
    by_file = test_analysis["by_file"]
    if by_file:
        # Sort files by total duration
        sorted_files = sorted(by_file.items(), key=lambda x: x[1]["total_duration"], reverse=True)[
            :10
        ]  # Show top 10 files

        print(f"\nTOP 10 TEST FILES BY TOTAL DURATION")
        print(
            "┌────────────────────────────────────────┬────────┬──────────┬──────────┬──────────┐"
        )
        print(
            "│ Test File                              │ Tests  │ Total    │ Average  │ Slowest  │"
        )
        print(
            "├────────────────────────────────────────┼────────┼──────────┼──────────┼──────────┤"
        )

        for file_name, file_data in sorted_files:
            # Truncate long file names
            display_name = file_name if len(file_name) <= 38 else "..." + file_name[-35:]
            test_count = file_data["test_count"]
            total_dur = format_duration(int(file_data["total_duration"]))
            avg_dur = f"{file_data['average_duration']:.1f}s"
            slowest_dur = (
                f"{file_data['slowest_test']['duration']:.1f}s"
                if file_data["slowest_test"]
                else "N/A"
            )

            print(
                f"│ {display_name:<38} │ {test_count:<6} │ {total_dur:<8} │ {avg_dur:<8} │ {slowest_dur:<8} │"
            )

        print(
            "└────────────────────────────────────────┴────────┴──────────┴──────────┴──────────┘"
        )

    # Top N slowest tests
    if top_n > 0 and test_analysis["all_tests"]:
        slowest_tests = test_analysis["all_tests"][:top_n]

        print(f"\nTOP {min(top_n, len(test_analysis['all_tests']))} SLOWEST TESTS")
        print(
            "┌────────────────────────────────────────────────────────────────┬──────────┬─────────────┐"
        )
        print(
            "│ Test Name                                                      │ Duration │ Job-Group   │"
        )
        print(
            "├────────────────────────────────────────────────────────────────┼──────────┼─────────────┤"
        )

        for test in slowest_tests:
            # Extract just the test method name for display
            test_parts = test["name"].split("::")
            if len(test_parts) > 1:
                display_name = "::".join(test_parts[-2:]) if len(test_parts) > 2 else test_parts[-1]
            else:
                display_name = test["name"]

            # Truncate if too long
            if len(display_name) > 62:
                display_name = "..." + display_name[-59:]

            duration = format_duration(int(test["duration"]))
            job_group = f"{test['job']}-{test['group']}"
            if len(job_group) > 11:
                job_group = job_group[:11]

            print(f"│ {display_name:<62} │ {duration:<8} │ {job_group:<11} │")

        print(
            "└────────────────────────────────────────────────────────────────┴──────────┴─────────────┘"
        )

    # Test distribution by group
    by_group = test_analysis["by_group"]
    if by_group:
        print(f"\nTEST DISTRIBUTION BY GROUP")
        print("┌─────────────────────┬──────────┬────────────┬──────────────┐")
        print("│ Job-Group           │ Tests    │ Total Time │ Avg Duration │")
        print("├─────────────────────┼──────────┼────────────┼──────────────┤")

        # Sort groups by total duration
        sorted_groups = sorted(
            by_group.items(), key=lambda x: x[1]["total_duration"], reverse=True
        )[:15]  # Show top 15 groups

        for group_key, group_data in sorted_groups:
            display_key = group_key if len(group_key) <= 19 else group_key[:19]
            test_count = group_data["test_count"]
            total_dur = format_duration(int(group_data["total_duration"]))
            avg_dur = (
                f"{group_data['total_duration'] / test_count:.1f}s" if test_count > 0 else "0.0s"
            )

            print(f"│ {display_key:<19} │ {test_count:<8} │ {total_dur:<10} │ {avg_dur:<12} │")

        print("└─────────────────────┴──────────┴────────────┴──────────────┘")


def print_human_readable(
    run_info: dict[str, str],
    analysis: dict[str, dict[str, str]],
    test_analysis: dict[str, Any] | None = None,
    top_n: int = 20,
):
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
                # Clean up common step names for better readability
                if "/.github/actions/free-disk-space" in step_name:
                    step_display = "Free disk space"
                elif "/.github/actions/setup-python" in step_name:
                    step_display = "Setup Python"
                elif "/.github/actions/setup-pyenv" in step_name:
                    step_display = "Setup pyenv"
                elif "/.github/actions/setup-java" in step_name:
                    step_display = "Setup Java"
                elif "actions/checkout" in step_name:
                    step_display = "Checkout"
                elif "Post Run" in step_name:
                    step_display = "Post: " + step_name.split("Post Run ")[-1][:18]
                elif step_name.startswith("Run "):
                    # Extract meaningful part after "Run "
                    remainder = step_name[4:]
                    if "/.github/actions/" in remainder:
                        # Extract action name
                        action_name = remainder.split("/")[-1][:20]
                        step_display = f"Run {action_name}"
                    else:
                        step_display = step_name[:24]
                else:
                    step_display = step_name[:24]  # Truncate long names

                min_dur = format_duration(int(stats["min"]))
                max_dur = format_duration(int(stats["max"]))
                avg_dur = format_duration(int(stats["average"]))
                print(
                    f"│ {step_display:<24} │ {min_dur:<7} │ {max_dur:<7} "
                    f"│ {avg_dur:<7} │ {stats['variance']:<8.1f} │"
                )

        print("└──────────────────────────┴─────────┴─────────┴─────────┴──────────┘")

    # Print test analysis if available
    if test_analysis:
        print_test_analysis(test_analysis, top_n)


def print_json_output(
    run_info: dict[str, str], analysis: dict[str, dict[str, str]], test_analysis: dict[str, Any] | None = None
):
    """Print analysis in JSON format."""
    output = {
        "run_id": run_info.get("id"),
        "run_url": run_info.get("html_url"),
        "status": run_info.get("status"),
        "conclusion": run_info.get("conclusion"),
        "jobs": analysis,
    }

    # Include test analysis if available
    if test_analysis:
        output["test_analysis"] = test_analysis

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
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of slowest tests to show in table view (default: 20, use 0 for none)",
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

    # Download and analyze test durations
    test_data = download_test_durations(run_id, repo)
    test_analysis = None
    if test_data:
        test_analysis = analyze_test_durations(test_data)
    else:
        print("\nNo test duration artifacts found in this run")

    # Output results
    if args.json:
        print_json_output(run_info, analysis, test_analysis)
    else:
        print_human_readable(run_info, analysis, test_analysis, args.top_n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
