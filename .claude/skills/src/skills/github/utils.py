# ruff: noqa: T201
import os
import re
import subprocess
import sys


def get_github_token() -> str:
    if token := os.environ.get("GITHUB_TOKEN"):
        return token
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: GITHUB_TOKEN not found (set env var or install gh CLI)", file=sys.stderr)
        sys.exit(1)


def parse_pr_url(url: str) -> tuple[str, str, int]:
    if m := re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url):
        return m.group(1), m.group(2), int(m.group(3))
    raise ValueError(f"Invalid PR URL: {url}")
