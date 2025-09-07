#!/usr/bin/env python
"""
Download and merge test duration data from MLflow CI runs.

This script is part of MLflow's intelligent test parallelization system. It downloads
test timing data collected by pytest-split during CI runs and merges it into
consolidated duration files used for future test splitting.

How it works:
1. MLflow CI jobs collect test execution times using pytest-split's --store-durations
2. Each job uploads timing data as artifacts (test-durations-{job}-group-{N})
3. This script downloads these artifacts from completed CI runs
4. The timing data is merged into consolidated files
   (.github/workflows/test_durations/{job}.test_duration)
5. Future CI runs use these consolidated files for intelligent test distribution

The result is a feedback loop where test execution times continuously improve
test parallelization, reducing overall CI runtime.

Usage:
    # Download from latest master run
    python dev/download_test_durations.py

    # Download from specific run ID
    python dev/download_test_durations.py --run-id 1234567890

    # Download from latest run on specific branch
    python dev/download_test_durations.py --branch my-feature-branch

    # Download from specific repo (default: mlflow/mlflow)
    python dev/download_test_durations.py --run-id 1234567890 --repo mlflow/mlflow
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Map job names to their configuration
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


def run_command(cmd, capture=True, check=True):
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


def get_github_repo():
    """Get GitHub repo - always returns mlflow/mlflow."""
    return "mlflow/mlflow"


def get_recent_run_id(repo, branch="master", status="completed"):
    """Get the most recent run ID for a branch."""
    cmd = (
        f"gh run list --repo {repo} --branch {branch} --status {status} "
        f'--limit 1 --json databaseId --jq ".[0].databaseId"'
    )
    run_id = run_command(cmd, check=False)
    if run_id and run_id.isdigit():
        return run_id
    return None


def download_duration_artifacts(run_id, repo):
    """Download all duration artifacts from a workflow run and organize by job."""
    job_durations = {}
    total_groups_found = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        print("\nDownloading artifacts to temporary directory...")

        for job_name, config in JOB_CONFIG.items():
            merged_durations = {}
            groups_found = 0

            for group in range(1, config["groups"] + 1):
                artifact_name = f"test-durations-{job_name}-group-{group}"

                # Try to download the artifact
                download_cmd = (
                    f"cd {tmpdir} && gh run download {run_id} --repo {repo} "
                    f"--name {artifact_name} 2>/dev/null"
                )
                result = run_command(download_cmd, check=False)
                # Check if download was successful (result is None on failure, stdout on success)
                if result is not None:
                    # The artifact is downloaded directly as group_N_durations.json
                    duration_file = Path(tmpdir) / f"group_{group}_durations.json"
                    if duration_file.exists():
                        groups_found += 1
                        with open(duration_file) as f:
                            group_durations = json.load(f)
                        if isinstance(group_durations, dict):
                            merged_durations.update(group_durations)
                        # Clean up to avoid conflicts with next download
                        duration_file.unlink()

            if groups_found > 0:
                job_durations[job_name] = merged_durations
                total_groups_found += groups_found
                print(
                    f"  {job_name}: Found {groups_found}/{config['groups']} groups, "
                    f"{len(merged_durations)} total durations"
                )
            else:
                print(f"  {job_name}: No artifacts found")

    return job_durations, total_groups_found


def save_job_durations(job_durations):
    """Save duration data to per-job files in .github/workflows/test_durations/."""
    durations_dir = Path(".github/workflows/test_durations")
    durations_dir.mkdir(parents=True, exist_ok=True)

    for job_name, durations in job_durations.items():
        target_file = durations_dir / f"{job_name}.test_duration"
        with open(target_file, "w") as f:
            json.dump(durations, f, indent=2, sort_keys=True)
        print(f"✓ Updated {target_file} with {len(durations)} durations")


def main():
    parser = argparse.ArgumentParser(
        description="Download and merge test duration data from MLflow CI runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from latest master run
  python dev/download_test_durations.py

  # Download from specific run ID
  python dev/download_test_durations.py --run-id 1234567890

  # Download from latest run on specific branch
  python dev/download_test_durations.py --branch my-feature-branch

  # Download from specific repo
  python dev/download_test_durations.py --run-id 1234567890 --repo mlflow/mlflow
""",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="GitHub Actions run ID to download artifacts from",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="master",
        help="Branch name to get latest run from (default: master)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="GitHub repository (e.g. mlflow/mlflow). Defaults to current repo or mlflow/mlflow",
    )

    args = parser.parse_args()
    repo = args.repo if args.repo else get_github_repo()

    # Determine which run to download from
    if args.run_id:
        run_id = args.run_id
        print(f"Using specified run ID: {run_id}")
    else:
        print(f"Finding latest run for branch: {args.branch}")
        run_id = get_recent_run_id(repo, args.branch)
        if not run_id:
            print(f"Error: No successful runs found for branch {args.branch}")
            return 1
        print(f"Found run ID: {run_id}")

    print(f"View run at: https://github.com/{repo}/actions/runs/{run_id}")

    # Download and merge artifacts
    job_durations, total_groups_found = download_duration_artifacts(run_id, repo)

    if not job_durations:
        print("\nNo duration data found in any job")
        return 1

    print(
        f"\nCollected durations from {len(job_durations)} jobs, {total_groups_found} total groups"
    )

    # Save to per-job files
    save_job_durations(job_durations)

    print("\n✓ Duration collection complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
