import argparse
import concurrent.futures
import itertools
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from packaging.version import Version


def get_token() -> str | None:
    if token := os.environ.get("GH_TOKEN"):
        return token
    try:
        token = subprocess.check_output(
            ["gh", "auth", "token"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        if token:
            return token
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


def get_headers() -> dict[str, str]:
    if token := get_token():
        return {"Authorization": f"token {token}"}
    return {}


def validate_version(version: str) -> None:
    """
    Validate that the version has a micro version component.
    Raises ValueError if the version is invalid.
    """
    parsed_version = Version(version)
    if len(parsed_version.release) != 3:
        raise ValueError(
            f"Invalid version: '{version}'. "
            "Version must be in the format <major>.<minor>.<micro> (e.g., '2.10.0')"
        )


def get_release_branch(version: str) -> str:
    major_minor_version = ".".join(version.split(".")[:2])
    return f"branch-{major_minor_version}"


@dataclass(frozen=True)
class Commit:
    sha: str
    pr_num: int
    date: str


def get_commit_count(branch: str, since: str) -> int:
    """
    Get the total count of commits in the branch since the given date using GraphQL API.
    """
    query = """
    query($branch: String!, $since: GitTimestamp!) {
      repository(owner: "mlflow", name: "mlflow") {
        ref(qualifiedName: $branch) {
          target {
            ... on Commit {
              history(since: $since) {
                totalCount
              }
            }
          }
        }
      }
    }
    """
    response = requests.post(
        "https://api.github.com/graphql",
        json={"query": query, "variables": {"branch": branch, "since": since}},
        headers=get_headers(),
    )
    response.raise_for_status()
    data = response.json()
    ref = data["data"]["repository"]["ref"]
    if ref is None:
        raise ValueError(f"Branch '{branch}' not found")
    total_count: int = ref["target"]["history"]["totalCount"]
    return total_count


def get_commits(branch: str) -> list[Commit]:
    """
    Get the commits in the release branch via GitHub API (last 90 days).
    Returns commits sorted by date (oldest first).
    """
    per_page = 100
    pr_rgx = re.compile(r".+\s+\(#(\d+)\)$")
    since = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()

    # Get total commit count first
    total_count = get_commit_count(branch, since)
    if total_count == 0:
        print(f"No commits found in {branch} since {since}")
        return []

    total_pages = (total_count + per_page - 1) // per_page
    print(f"Total commits: {total_count}, fetching {total_pages} page(s)...")

    def fetch_page(page: int) -> list[Commit]:
        print(f"Fetching page {page}/{total_pages}...")
        params: dict[str, str | int] = {
            "sha": branch,
            "per_page": per_page,
            "page": page,
            "since": since,
        }
        response = requests.get(
            "https://api.github.com/repos/mlflow/mlflow/commits",
            params=params,
            headers=get_headers(),
        )
        response.raise_for_status()
        commits = []
        for item in response.json():
            msg = item["commit"]["message"].split("\n")[0]
            if m := pr_rgx.search(msg):
                # Use committer date (not author date) because cherry-picked commits
                # retain the original author date but get a new committer date.
                date = item["commit"]["committer"]["date"]
                commits.append(Commit(sha=item["sha"], pr_num=int(m.group(1)), date=date))
        return commits

    # Fetch all pages in parallel. executor.map preserves order.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(fetch_page, range(1, total_pages + 1))

    return sorted(itertools.chain.from_iterable(results), key=lambda c: c.date)


@dataclass(frozen=True)
class PR:
    pr_num: int
    merged: bool


def is_closed(pr: dict[str, Any]) -> bool:
    return pr["state"] == "closed" and pr["pull_request"]["merged_at"] is None


def fetch_patch_prs(version: str) -> dict[int, bool]:
    """
    Fetch PRs labeled with `v{version}` from the MLflow repository.
    """
    label = f"v{version}"
    per_page = 100
    page = 1
    pulls: list[dict[str, Any]] = []
    while True:
        response = requests.get(
            f'https://api.github.com/search/issues?q=is:pr+repo:mlflow/mlflow+label:"{label}"&per_page={per_page}&page={page}',
            headers=get_headers(),
        )
        response.raise_for_status()
        data = response.json()
        # Exclude closed PRs that are not merged
        pulls.extend(pr for pr in data["items"] if not is_closed(pr))
        if len(data["items"]) < per_page:
            break
        page += 1

    return {pr["number"]: pr["pull_request"].get("merged_at") is not None for pr in pulls}


def main(version: str, dry_run: bool) -> None:
    validate_version(version)
    release_branch = get_release_branch(version)
    commits = get_commits(release_branch)
    patch_prs = fetch_patch_prs(version)
    if not_cherry_picked := set(patch_prs) - {c.pr_num for c in commits}:
        print(f"The following patch PRs are not cherry-picked to {release_branch}:")
        for idx, pr_num in enumerate(sorted(not_cherry_picked)):
            merged = patch_prs[pr_num]
            url = f"https://github.com/mlflow/mlflow/pull/{pr_num} (merged: {merged})"
            line = f"  {idx + 1}. {url}"
            if not merged:
                line = f"\033[91m{line}\033[0m"  # Red color using ANSI escape codes
            print(line)

        master_commits = get_commits("master")
        cherry_picks = [c.sha for c in master_commits if c.pr_num in not_cherry_picked]
        print("\n# Steps to cherry-pick the patch PRs:")
        print(
            f"1. Make sure your local master and {release_branch} branches are synced with "
            "upstream."
        )
        print(f"2. Cut a new branch from {release_branch} (e.g. {release_branch}-cherry-picks).")
        print("3. Run the following command on the new branch:\n")
        print("git cherry-pick " + " ".join(cherry_picks))
        print(f"\n4. File a PR against {release_branch}.")
        sys.exit(0 if dry_run else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="The version to release")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.environ.get("DRY_RUN", "true").lower() == "true",
        help="Dry run mode (default: True, can be set via DRY_RUN env var)",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Disable dry run mode",
    )
    args = parser.parse_args()
    main(args.version, args.dry_run)
