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

import argparse
import os
import subprocess


def main(new_version: str, remote: str, dry_run: bool = False):
    release_tag = f"v{new_version}"
    subprocess.run(["git", "tag", release_tag], check=True)
    subprocess.run(
        ["git", "push", remote, release_tag, *(["--dry-run"] if dry_run else [])], check=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a release tag")
    parser.add_argument("--new-version", required=True, help="New version to release")
    parser.add_argument("--remote", default="origin", help="Git remote to use (default: origin)")
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
    main(args.new_version, args.remote, args.dry_run)
