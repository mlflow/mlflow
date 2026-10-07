# ruff: noqa: T201
import os
import subprocess
import sys


def resolve_github_token() -> str | None:
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
