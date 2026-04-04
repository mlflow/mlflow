"""
Pre-commit hook to validate GitHub Action pins in workflow and action files.

Ensures that every remote GitHub Action reference:
1. Uses a full 40-character SHA pin (not a tag or branch).
2. Has a trailing ``# vX.Y.Z`` version comment documenting the pinned tag.
3. The SHA matches the claimed tag (verified via GitHub API, results cached).

Usage:
    uv run dev/check_action_pins.py [file ...]

When file paths are provided (pre-commit mode), only those files are checked.
When no arguments are given, all .github/workflows/ and .github/actions/ YAML
files are scanned.

Cache:
    Verification results are stored in .cache/action-pins.json to avoid
    redundant network calls across repeated runs.
"""

import functools
import glob
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Matches a `uses:` line that references a remote action (not a local `./` path).
# Captures:  owner/repo[/subpath]  @  ref  [  # comment  ]
_USES_RE = re.compile(
    r"""
    ^\s*-?\s*uses:\s+          # leading `- uses:` or `uses:`
    (?P<action>[^@\s]+)        # owner/repo[/subpath]
    @
    (?P<ref>[^\s#]+)           # ref (SHA, tag, or branch)
    (?:\s+\#\s*(?P<comment>\S+))?  # optional  # comment
    """,
    re.VERBOSE,
)

# A full 40-character hexadecimal SHA.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Requires at least vMAJOR.MINOR.PATCH to avoid ambiguous moving tags like v4.
_VERSION_COMMENT_RE = re.compile(r"^v\d+\.\d+\.\d+(?:\.\d+)*$")

_CACHE_PATH = Path(".cache/action-pins.json")
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0  # seconds


def _load_cache() -> dict[str, bool]:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text())  # type: ignore[no-any-return]
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: dict[str, bool]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))
    except OSError:
        pass


@functools.cache
def _get_github_token() -> str | None:
    if token := os.environ.get("GH_TOKEN"):
        return token
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True).strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _github_api(url: str) -> dict[str, Any] | list[Any] | None:

    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token := _get_github_token():
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())  # type: ignore[no-any-return]
        except urllib.error.HTTPError as e:
            if e.code in (404, 401, 403):
                return None
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
        except urllib.error.URLError:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
    return None


def _repo_from_action(action: str) -> str:
    match action.split("/"):
        case [owner, repo, *_]:
            return f"{owner}/{repo}"
        case _:
            raise ValueError(f"Invalid action format: {action!r}")


def _verify_sha_tag(action: str, sha: str, tag: str, cache: dict[str, bool]) -> bool | None:

    cache_key = f"{action}@{sha}#{tag}"
    if cache_key in cache:
        return cache[cache_key]

    repo = _repo_from_action(action)
    url = f"https://api.github.com/repos/{repo}/git/ref/tags/{tag}"
    data = _github_api(url)

    match data:
        case {"object": {"type": "commit", "sha": str(commit_sha)}}:
            result = commit_sha == sha
        case {"object": {"type": "tag", "sha": str(tag_sha)}}:
            tag_url = f"https://api.github.com/repos/{repo}/git/tags/{tag_sha}"
            match _github_api(tag_url):
                case {"object": {"sha": str(commit_sha)}}:
                    result = commit_sha == sha
                case _:
                    return None
        case None:
            return None
        case _:
            result = False

    cache[cache_key] = result
    _save_cache(cache)
    return result


def _iter_files(args: list[str]) -> Iterator[Path]:
    if args:
        yield from (Path(a) for a in args)
    else:
        for pattern in (
            ".github/workflows/*.yml",
            ".github/workflows/*.yaml",
            ".github/actions/**/*.yml",
            ".github/actions/**/*.yaml",
        ):
            yield from (Path(p) for p in glob.glob(pattern, recursive=True))


def check_file(path: Path, cache: dict[str, bool]) -> list[str]:

    errors = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        return [f"{path}: cannot read file: {e}"]

    for lineno, line in enumerate(lines, start=1):
        m = _USES_RE.match(line)
        if m is None:
            continue
        action = m.group("action")
        ref = m.group("ref")
        comment = m.group("comment")

        # Skip local composite actions.
        if action.startswith("./"):
            continue

        prefix = f"{path}:{lineno}: {line.strip()!r}"

        if not _SHA_RE.match(ref):
            errors.append(f"{prefix}\n  error: ref '{ref}' is not a 40-character SHA")
            continue

        if not comment or not _VERSION_COMMENT_RE.match(comment):
            errors.append(
                f"{prefix}\n  error: missing or invalid version comment"
                f" (expected '# vX.Y.Z', got {comment!r})"
            )
            continue

        verified = _verify_sha_tag(action, ref, comment, cache)
        if verified is None:
            errors.append(
                f"{prefix}\n  error: could not verify SHA against tag '{comment}'"
                f" for {_repo_from_action(action)} (GitHub API unavailable)"
            )
        elif not verified:
            errors.append(
                f"{prefix}\n  error: SHA '{ref}' does not match tag '{comment}'"
                f" for {_repo_from_action(action)}"
            )

    return errors


def main() -> int:
    args = sys.argv[1:]
    cache = _load_cache()
    all_errors: list[str] = []
    for path in _iter_files(args):
        all_errors.extend(check_file(path, cache))

    if all_errors:
        print("action-pins: the following violations were found:\n")
        for err in all_errors:
            print(err)
        print(f"\n{len(all_errors)} violation(s) found.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
