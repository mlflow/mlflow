# ruff: noqa: T201
"""Upload media to GitHub's user-attachments store and print the URL for each file."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from skills.github.uploads import UploadFailed, upload_asset
from skills.github.utils import get_github_token

DEFAULT_REPO = "mlflow/mlflow"


def resolve_repository_id(repo: str) -> str:
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}", "--jq", ".id"],
            capture_output=True,
            text=True,
            check=True,
        )
    # `gh` writes the actionable part ("Not Found (HTTP 404)", an auth error) to
    # stderr, which CalledProcessError leaves out of its own message.
    except subprocess.CalledProcessError as e:
        print(f"Could not resolve {repo}: {e.stderr.strip() or e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Could not resolve {repo}: {e}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "upload-media",
        help="Upload images or videos and print their user-attachments URLs",
    )
    parser.add_argument("paths", type=Path, nargs="+", help="Media files to upload")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Repository the assets are bound to (default: {DEFAULT_REPO})",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    token = get_github_token()
    repository_id = resolve_repository_id(args.repo)

    failed = False
    for path in args.paths:
        if not path.is_file():
            print(f"failed {path}: not a file", file=sys.stderr)
            failed = True
            continue
        try:
            print(f"{path}\t{upload_asset(path, repository_id, token)}")
        except UploadFailed as e:
            failed = True
            # A fault that is not about this file fails every remaining upload, so
            # stop asking, and name the credential this command resolved.
            if e.fatal:
                hint = "; check GH_TOKEN or run `gh auth login`" if e.status == 401 else ""
                print(f"failed {e}{hint}", file=sys.stderr)
                break
            print(f"failed {e}", file=sys.stderr)

    if failed:
        sys.exit(1)
