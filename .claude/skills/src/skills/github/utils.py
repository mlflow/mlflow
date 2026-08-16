# ruff: noqa: T201
import os
import re
import subprocess
import sys


def resolve_github_token() -> str | None:
    """Return a token, or None for callers that must keep going without one.

    Read from the environment, never argv: a PAT in a CLI argument is visible in the
    process list for the life of the call.
    """
    if token := os.environ.get("GH_TOKEN"):
        return token
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_github_token() -> str:
    if token := resolve_github_token():
        return token
    print("Error: GH_TOKEN not found (set env var or install gh CLI)", file=sys.stderr)
    sys.exit(1)


def parse_pr_url(url: str) -> tuple[str, str, int]:
    if m := re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url):
        return m.group(1), m.group(2), int(m.group(3))
    raise ValueError(f"Invalid PR URL: {url}")
