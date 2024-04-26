import re
import subprocess
import sys
from dataclasses import dataclass

import click
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
    log_stdout = subprocess.check_output(
        [
            "git",
            "log",
            branch,
            '--since="3 months ago"',
            "--pretty=format:%H %s",
        ],
        text=True,
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


def fetch_patch_prs(version):
    """
    Fetch PRs labeled with `patch-{version}` from the MLflow repository.
    """
    label = f"patch-{version}"
    per_page = 100
    page = 1
    pulls = []
    while True:
        response = requests.get(
            f'https://api.github.com/search/issues?q=is:pr+repo:mlflow/mlflow+label:"{label}"&per_page={per_page}&page={page}',
        )
        response.raise_for_status()
        data = response.json()
        pulls.extend(data["items"])
        if len(data) < per_page:
            break
        page += 1

    return {pr["number"]: pr["pull_request"].get("merged_at") is not None for pr in pulls}


@click.command()
@click.option("--version", required=True, help="The version to release")
def main(version):
    release_branch = get_release_branch(version)
    commits = get_commits(release_branch)
    patch_prs = fetch_patch_prs(version)
    if not_cherry_picked := set(patch_prs) - {c.pr_num for c in commits}:
        click.echo(f"The following patch PRs are not cherry-picked to {release_branch}:")
        for idx, pr_num in enumerate(sorted(not_cherry_picked)):
            url = f"https://github.com/mlflow/mlflow/pull/{pr_num} (merged: {patch_prs[pr_num]})"
            click.echo(f"  {idx + 1}. {url}")

        master_commits = get_commits("master")
        cherry_picks = [c.sha for c in master_commits if c.pr_num in not_cherry_picked]
        # reverse the order of cherry-picks to maintain the order of PRs
        print("\nTo cherry-pick the above commits, run:")
        print("git cherry-pick " + " ".join(cherry_picks[::-1]))
        sys.exit(1)


if __name__ == "__main__":
    main()
