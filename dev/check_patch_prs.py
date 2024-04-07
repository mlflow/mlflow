import re
import subprocess
import sys

import click
import requests


def get_release_branch(version):
    major_minor_version = ".".join(version.split(".")[:2])
    return f"branch-{major_minor_version}"


def get_merged_prs(version):
    """
    Get the PRs merged into the release branch for the given version.
    """
    release_branch = get_release_branch(version)
    commits = subprocess.check_output(
        [
            "git",
            "log",
            release_branch,
            '--since="2 months ago"',
            "--pretty=format:%s",
        ],
        text=True,
    )
    pr_rgx = re.compile(r"\s+\(#(\d+)\)$")
    prs = set()
    for commit in commits.splitlines():
        if pr := pr_rgx.search(commit.rstrip()):
            prs.add(int(pr.group(1)))

    return prs


def fetch_patch_prs(version):
    """
    Fetch PRs labeled with `patch-{version}` from the MLflow repository.
    """
    label = f"patch-{version}"
    response = requests.get(
        f'https://api.github.com/search/issues?q=is:pr+repo:mlflow/mlflow+label:"{label}"'
    )
    response.raise_for_status()
    data = response.json()
    return {pr["number"] for pr in data["items"]}


@click.command()
@click.option("--version", required=True, help="The version to release")
def main(version):
    prs = get_merged_prs(version)
    patch_prs = fetch_patch_prs(version)
    if not_cherry_picked := patch_prs - prs:
        branch = get_release_branch(version)
        click.echo(f"The following patch PRs are not cherry-picked to {branch}:")
        for pr_num in sorted(not_cherry_picked):
            url = f"https://github.com/mlflow/mlflow/pull/{pr_num}"
            click.echo(f"  - {url}")
        sys.exit(1)


if __name__ == "__main__":
    main()
