import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import requests


def get_release_branch(version):
    major_minor_version = ".".join(version.split(".")[:2])
    return f"branch-{major_minor_version}"


@dataclass(frozen=True)
class Commit:
    sha: str
    pr_num: int


def get_commits(branch: str):
    """
    Get the commits in the release branch.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.check_call(
            [
                "git",
                "clone",
                "--shallow-since=3 months ago",
                "--branch",
                branch,
                "https://github.com/mlflow/mlflow.git",
                tmpdir,
            ],
        )
        log_stdout = subprocess.check_output(
            [
                "git",
                "log",
                "--pretty=format:%H %s",
            ],
            text=True,
            cwd=tmpdir,
        )
        pr_rgx = re.compile(r"([a-z0-9]+) .+\s+\(#(\d+)\)$")
        commits = []
        for commit in log_stdout.splitlines():
            if m := pr_rgx.search(commit.rstrip()):
                commits.append(Commit(sha=m.group(1), pr_num=int(m.group(2))))

    return commits


@dataclass(frozen=True)
class PR:
    pr_num: int
    merged: bool


def is_closed(pr):
    return pr["state"] == "closed" and pr["pull_request"]["merged_at"] is None


def fetch_patch_prs(version):
    """
    Fetch PRs labeled with `v{version}` from the MLflow repository.
    """
    label = f"v{version}"
    per_page = 100
    page = 1
    pulls = []
    while True:
        response = requests.get(
            f'https://api.github.com/search/issues?q=is:pr+repo:mlflow/mlflow+label:"{label}"&per_page={per_page}&page={page}',
        )
        response.raise_for_status()
        data = response.json()
        # Exclude closed PRs that are not merged
        pulls.extend(pr for pr in data["items"] if not is_closed(pr))
        if len(data) < per_page:
            break
        page += 1

    return {pr["number"]: pr["pull_request"].get("merged_at") is not None for pr in pulls}


def main(version, dry_run):
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
        # reverse the order of cherry-picks to maintain the order of PRs
        print("\n# Steps to cherry-pick the patch PRs:")
        print(
            f"1. Make sure your local master and {release_branch} branches are synced with "
            "upstream."
        )
        print(f"2. Cut a new branch from {release_branch} (e.g. {release_branch}-cherry-picks).")
        print("3. Run the following command on the new branch:\n")
        print("git cherry-pick " + " ".join(cherry_picks[::-1]))
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
