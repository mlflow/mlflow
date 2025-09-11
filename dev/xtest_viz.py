# /// script
# dependencies = [
#     "pandas",
#     "tabulate",
# ]
# requires-python = ">=3.10"
# ///
"""
Script to visualize cross-version test results for MLflow autologging and models.

This script fetches scheduled workflow run results from GitHub Actions and generates
a markdown table showing the test status for different package versions across
different dates.

Usage:
    uv run dev/xtest_viz.py                      # Fetch last 14 days from mlflow/dev
    uv run dev/xtest_viz.py --days 30            # Fetch last 30 days
    uv run dev/xtest_viz.py --repo mlflow/mlflow  # Use different repo

Example output (truncated for brevity):
    | Name                                   | 2024-01-15 | 2024-01-14 | 2024-01-13 |
    |----------------------------------------|------------|------------|------------|
    | test1 (sklearn, 1.3.1, autologging...) | [✅](link) | [✅](link) | [❌](link) |
    | test1 (pytorch, 2.1.0, models...)      | [✅](link) | [⚠️](link) | [✅](link) |
    | test2 (xgboost, 2.0.0, autologging...) | [❌](link) | [✅](link) | —          |

Where:
    ✅ = success
    ❌ = failure
    ⚠️ = cancelled
    ❓ = unknown status
    — = no data
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

import pandas as pd


class XTestViz:
    def __init__(self, github_token: str | None = None, repo: str = "mlflow/dev"):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.repo = repo
        self.per_page = 30  # Items per page for API requests
        self.headers: dict[str, str] = {}
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"
            self.headers["Accept"] = "application/vnd.github.v3+json"

    def _make_request(self, url: str) -> dict[str, Any]:
        """Make an HTTP GET request and return JSON response."""
        req = urllib.request.Request(url, headers=self.headers)
        try:
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            error_msg = f"HTTP {e.code}: {e.reason}"
            if e.code == 401:
                error_msg += " (Check your GitHub token)"
            elif e.code == 404:
                error_msg += " (Repository or workflow not found)"
            raise Exception(error_msg) from e

    def get_workflow_runs(self, days_back: int = 30) -> list[dict[str, Any]]:
        """Fetch cross-version test workflow runs from the last N days."""
        since_date = (datetime.now() - timedelta(days=days_back)).isoformat()

        print(f"Fetching scheduled workflow runs from last {days_back} days...", file=sys.stderr)

        all_runs: list[dict[str, Any]] = []
        page: int = 1

        while True:
            params = {
                "per_page": self.per_page,
                "page": page,
                "created": f">={since_date}",
                "status": "completed",
                "event": "schedule",  # Only fetch scheduled runs
            }
            query_string = urllib.parse.urlencode(params)
            url = f"https://api.github.com/repos/{self.repo}/actions/workflows/cross-version-tests.yml/runs?{query_string}"

            data = self._make_request(url)
            runs = data.get("workflow_runs", [])

            if not runs:
                break

            all_runs.extend(runs)

            print(f"  Fetched page {page} ({len(runs)} runs)", file=sys.stderr)

            # Check if there are more pages
            if len(runs) < self.per_page:
                break

            page += 1

        print(f"Found {len(all_runs)} scheduled workflow runs total", file=sys.stderr)

        return all_runs

    def get_workflow_jobs(self, run_id: int) -> list[dict[str, Any]]:
        """Get jobs for a specific workflow run."""
        all_jobs: list[dict[str, Any]] = []
        page: int = 1

        while True:
            params = {"per_page": self.per_page, "page": page}
            query_string = urllib.parse.urlencode(params)
            url = f"https://api.github.com/repos/{self.repo}/actions/runs/{run_id}/jobs?{query_string}"

            data = self._make_request(url)
            jobs = data.get("jobs", [])

            if not jobs:
                break

            all_jobs.extend(jobs)

            # Check if there are more pages
            if len(jobs) < self.per_page:
                break

            page += 1

        return all_jobs

    def _fetch_run_jobs(self, run: dict[str, Any]) -> list[dict[str, Any]]:
        """Fetch jobs for a single workflow run."""
        run_id = run["id"]
        run_date = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")).strftime(
            "%Y-%m-%d"
        )

        jobs = self.get_workflow_jobs(run_id)
        data_rows = []

        for job in jobs:
            # Determine status emoji and link
            status = job["conclusion"]
            if status == "success":
                emoji = "✅"
            elif status == "failure":
                emoji = "❌"
            elif status == "cancelled":
                emoji = "⚠️"
            elif status == "skipped":
                continue  # Skip skipped jobs
            else:
                emoji = "❓"

            job_url = job["html_url"]
            status_link = f"[{emoji}]({job_url})"

            data_rows.append({"Name": job["name"], "Date": run_date, "Status": status_link})

        return data_rows

    def generate_results_table(self, days_back: int = 30) -> str:
        """Generate markdown table of cross-version test results."""
        # Get workflow runs
        workflow_runs = self.get_workflow_runs(days_back)

        if not workflow_runs:
            return "No workflow runs found in the specified time period."

        # Collect all data using parallel fetching
        print(
            f"Fetching jobs for {len(workflow_runs)} workflow runs in parallel...", file=sys.stderr
        )
        data_rows = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all job fetching tasks
            future_to_run = {
                executor.submit(self._fetch_run_jobs, run): run for run in workflow_runs
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_run), 1):
                run = future_to_run[future]
                try:
                    run_data = future.result()
                    data_rows.extend(run_data)
                    print(
                        f"  Completed {i}/{len(workflow_runs)}: run {run['id']} "
                        f"({len(run_data)} jobs)",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(f"  Error fetching jobs for run {run['id']}: {e}", file=sys.stderr)

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        if df.empty:
            return "No test jobs found."

        # Pivot table to have dates as columns
        pivot_df = df.pivot_table(
            index="Name",
            columns="Date",
            values="Status",
            aggfunc="first",  # Use first value if duplicates exist
        )

        # Sort columns (dates) in descending order
        pivot_df = pivot_df[sorted(pivot_df.columns, reverse=True)]

        # Sort rows (test names) alphabetically
        pivot_df = pivot_df.sort_index()

        # Fill NaN values with em-dash
        pivot_df = pivot_df.fillna("—")

        # Reset index to make Name a regular column
        pivot_df = pivot_df.reset_index()

        # Convert to markdown
        return pivot_df.to_markdown(index=False, tablefmt="pipe")


def main():
    parser = argparse.ArgumentParser(description="Visualize MLflow cross-version test results")
    parser.add_argument(
        "--days", type=int, default=14, help="Number of days back to fetch results (default: 14)"
    )
    parser.add_argument(
        "--repo",
        default="mlflow/dev",
        help="GitHub repository in owner/repo format (default: mlflow/dev)",
    )
    parser.add_argument("--token", help="GitHub token (default: use GITHUB_TOKEN env var)")

    args = parser.parse_args()

    # Check for GitHub token
    token = args.token or os.getenv("GITHUB_TOKEN")
    if not token:
        print(
            "Warning: No GitHub token provided. API requests may be rate-limited.", file=sys.stderr
        )
        print("Set GITHUB_TOKEN environment variable or use --token option.", file=sys.stderr)

    visualizer = XTestViz(github_token=token, repo=args.repo)

    try:
        output = visualizer.generate_results_table(args.days)
        print(output)

    except urllib.error.URLError as e:
        print(f"Error fetching data from GitHub API: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
