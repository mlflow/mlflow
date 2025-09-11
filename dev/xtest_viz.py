# /// script
# dependencies = [
#     "pandas",
#     "tabulate",
#     "aiohttp",
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
import asyncio
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import pandas as pd


@dataclass
class JobResult:
    """Data class representing a test job result."""

    name: str
    package: str
    group: str
    version: str
    date: str
    status: str


class XTestViz:
    def __init__(self, github_token: str | None = None, repo: str = "mlflow/dev"):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.repo = repo
        self.per_page = 30  # Items per page for API requests
        self.headers: dict[str, str] = {}
        if self.github_token:
            self.headers["Authorization"] = f"token {self.github_token}"
            self.headers["Accept"] = "application/vnd.github.v3+json"

    def parse_job_name(self, job_name: str) -> tuple[str, str, str]:
        """Parse job name to extract package, group, and version.

        Expected format: "package (version, group)" or variations
        Examples:
        - "sklearn (1.3.1, autologging)"
        - "pytorch (2.1.0, models)"
        - "xgboost (2.0.0, autologging)"

        Returns:
            tuple: (package, group, version)
        """
        # Pattern to match: package_name (version, group)
        pattern = r"^([^(]+)\s*\(([^,]+),\s*([^)]+)\)"
        match = re.match(pattern, job_name.strip())

        if match:
            package = match.group(1).strip()
            version = match.group(2).strip()
            group = match.group(3).strip()
            return package, group, version

        # Fallback: if parsing fails, return the original name and empty values
        return job_name, "", ""

    async def _make_request(self, session: aiohttp.ClientSession, url: str) -> dict[str, Any]:
        """Make an async HTTP GET request and return JSON response."""
        async with session.get(url, headers=self.headers) as response:
            response.raise_for_status()
            return await response.json()

    async def get_workflow_runs(
        self, session: aiohttp.ClientSession, days_back: int = 30
    ) -> list[dict[str, Any]]:
        """Fetch cross-version test workflow runs from the last N days."""
        since_date = (datetime.now() - timedelta(days=days_back)).isoformat()

        print(f"Fetching scheduled workflow runs from last {days_back} days...", file=sys.stderr)

        all_runs: list[dict[str, Any]] = []
        page: int = 1

        while True:
            params = {
                "per_page": str(self.per_page),
                "page": str(page),
                "created": f">={since_date}",
                "status": "completed",
                "event": "schedule",  # Only fetch scheduled runs
            }
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"https://api.github.com/repos/{self.repo}/actions/workflows/cross-version-tests.yml/runs?{query_string}"

            data = await self._make_request(session, url)
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

    async def get_workflow_jobs(
        self, session: aiohttp.ClientSession, run_id: int
    ) -> list[dict[str, Any]]:
        """Get jobs for a specific workflow run."""
        all_jobs: list[dict[str, Any]] = []
        page: int = 1

        while True:
            params = {"per_page": str(self.per_page), "page": str(page)}
            query_string = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"https://api.github.com/repos/{self.repo}/actions/runs/{run_id}/jobs?{query_string}"

            data = await self._make_request(session, url)
            jobs = data.get("jobs", [])

            if not jobs:
                break

            all_jobs.extend(jobs)

            # Check if there are more pages
            if len(jobs) < self.per_page:
                break

            page += 1

        return all_jobs

    async def _fetch_run_jobs(
        self, session: aiohttp.ClientSession, run: dict[str, Any]
    ) -> list[JobResult]:
        """Fetch jobs for a single workflow run."""
        run_id = run["id"]
        run_date = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")).strftime(
            "%Y-%m-%d"
        )

        jobs = await self.get_workflow_jobs(session, run_id)
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

            # Parse job name to extract package, group, and version
            package, group, version = self.parse_job_name(job["name"])

            data_rows.append(
                JobResult(
                    name=job["name"],
                    package=package,
                    group=group,
                    version=version,
                    date=run_date,
                    status=status_link,
                )
            )

        return data_rows

    async def fetch_all_jobs(self, days_back: int = 30) -> list[JobResult]:
        """Fetch all jobs from workflow runs in the specified time period."""
        async with aiohttp.ClientSession() as session:
            # Get workflow runs
            workflow_runs = await self.get_workflow_runs(session, days_back)

            if not workflow_runs:
                return []

            # Collect all data using async parallel fetching
            print(
                f"Fetching jobs for {len(workflow_runs)} workflow runs concurrently...",
                file=sys.stderr,
            )

            # Create tasks for all workflow runs
            tasks = [self._fetch_run_jobs(session, run) for run in workflow_runs]

            # Execute all tasks concurrently and gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            data_rows = []

            for i, result in enumerate(results, 1):
                if isinstance(result, Exception):
                    print(f"  Error fetching jobs for run {i}: {result}", file=sys.stderr)
                else:
                    data_rows.extend(result)
                    print(
                        f"  Completed {i}/{len(workflow_runs)} ({len(result)} jobs)",
                        file=sys.stderr,
                    )

            return data_rows

    def render_results_table(self, data_rows: list[JobResult]) -> str:
        """Render job data as a markdown table."""
        if not data_rows:
            return "No test jobs found."

        # Convert dataclass instances to dictionaries for pandas
        df_data = [
            {
                "Package": row.package,
                "Group": row.group,
                "Version": row.version,
                "Date": row.date,
                "Status": row.status,
            }
            for row in data_rows
        ]
        df = pd.DataFrame(df_data)

        # Create a composite key for the index
        df["Test"] = df["Package"] + " (" + df["Version"] + ", " + df["Group"] + ")"

        # Pivot table to have dates as columns
        pivot_df = df.pivot_table(
            index="Test",
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

        # Reset index to make Test a regular column
        pivot_df = pivot_df.reset_index()

        # Convert to markdown
        return pivot_df.to_markdown(index=False, tablefmt="pipe")

    async def generate_results_table(self, days_back: int = 30) -> str:
        """Generate markdown table of cross-version test results."""
        data_rows = await self.fetch_all_jobs(days_back)
        if not data_rows:
            return "No workflow runs found in the specified time period."
        return self.render_results_table(data_rows)


async def main():
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
    output = await visualizer.generate_results_table(args.days)
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
