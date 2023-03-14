"""
How to test this script
-----------------------
# Ensure origin points to your fork
git remote -v | grep origin

# Pretend we're releasing MLflow 9.0.0
git checkout -b branch-9.0

# First, test the dry run mode
python dev/create_release_tag.py --new-version 9.0.0 --dry-run
git tag -d v9.0.0

# Open https://github.com/<username>/mlflow/tree/v9.0.0 and verify that the tag does not exist.

# Then, test the non-dry run mode
python dev/create_release_tag.py --new-version 9.0.0 --no-dry-run
git tag -d v9.0.0

# Open https://github.com/<username>/mlflow/tree/v9.0.0 and verify that the tag exists now.

# Clean up the remote tag
git push --delete origin v9.0.0

# Clean up the local release branch
git checkout master
git branch -D branch-9.0
"""
import subprocess

import click
from packaging.version import Version


@click.command(help="Create a release tag")
@click.option("--new-version", required=True)
@click.option("--remote", required=False, default="origin", show_default=True)
@click.option(
    "--dry-run/--no-dry-run", is_flag=True, default=True, show_default=True, envvar="DRY_RUN"
)
def main(new_version: str, remote: str, dry_run: bool = False):
    version = Version(new_version)
    release_branch = f"branch-{version.major}.{version.minor}"
    release_tag = f"v{new_version}"
    subprocess.check_call(["git", "tag", release_tag, release_branch])
    subprocess.check_call(["git", "push", remote, release_tag, *(["--dry-run"] if dry_run else [])])


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
