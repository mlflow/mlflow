import argparse
import os
import subprocess

from packaging.version import Version


def main(new_version: str, remote: str, dry_run=False):
    version = Version(new_version)
    release_branch = f"branch-{version.major}.{version.minor}"
    exists_on_remote = (
        subprocess.check_output(
            ["git", "ls-remote", "--heads", remote, release_branch], text=True
        ).strip()
        != ""
    )
    if exists_on_remote:
        print(f"{release_branch} already exists on {remote}, skipping branch creation")
        return

    prev_branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    try:
        exists_on_local = (
            subprocess.check_output(["git", "branch", "--list", release_branch], text=True).strip()
            != ""
        )
        if exists_on_local:
            print(f"Deleting existing {release_branch}")
            subprocess.check_call(["git", "branch", "-D", release_branch])

        print(f"Creating {release_branch}")
        subprocess.check_call(["git", "checkout", "-b", release_branch, "master"])
        print(f"Pushing {release_branch} to {remote}")
        subprocess.check_call(
            ["git", "push", remote, release_branch, *(["--dry-run"] if dry_run else [])]
        )
    finally:
        subprocess.check_call(["git", "checkout", prev_branch])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a release branch")
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
