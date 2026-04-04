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

import glob
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

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

# A version-comment that looks like a tag: v1, v1.2, v1.2.3 (with optional extra segments).
_VERSION_COMMENT_RE = re.compile(r"^v\d+(\.\d+)*$")

_CACHE_PATH = Path(".cache/action-pins.json")
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0  # seconds


def _load_cache() -> dict[str, bool]:
    if _CACHE_PATH.exists():
        try:
            data: dict[str, bool] = json.loads(_CACHE_PATH.read_text())
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache: dict[str, bool]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))
    except OSError:
        pass


def _github_api(url: str) -> dict[str, object] | list[object] | None:
    """Fetch a GitHub API URL, retrying up to _MAX_RETRIES times."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())  # type: ignore[no-any-return]
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
    return None


def _repo_from_action(action: str) -> str:
    """Return 'owner/repo' from an action string like 'owner/repo/subpath'."""
    parts = action.split("/")
    return "/".join(parts[:2])


def _verify_sha_tag(action: str, sha: str, tag: str, cache: dict[str, bool]) -> bool:
    """Return True if *sha* resolves to *tag* in the action's repo."""
    cache_key = f"{action}@{sha}#{tag}"
    if cache_key in cache:
        return cache[cache_key]

    repo = _repo_from_action(action)
    url = f"https://api.github.com/repos/{repo}/git/ref/tags/{tag}"
    data = _github_api(url)

    result = False
    if isinstance(data, dict):
        obj = data.get("object") or {}
        if not isinstance(obj, dict):
            obj = {}
        obj_type = obj.get("type")
        obj_sha = obj.get("sha")
        if obj_type == "commit":
            result = obj_sha == sha
        elif obj_type == "tag":
            # Annotated tag — need to dereference the tag object to get the commit SHA.
            tag_url = f"https://api.github.com/repos/{repo}/git/tags/{obj_sha}"
            tag_data = _github_api(tag_url)
            if isinstance(tag_data, dict):
                inner = tag_data.get("object") or {}
                if isinstance(inner, dict):
                    result = inner.get("sha") == sha

    cache[cache_key] = result
    _save_cache(cache)
    return result


def _collect_files(args: list[str]) -> list[Path]:
    if args:
        return [Path(a) for a in args]
    patterns = [
        ".github/workflows/*.yml",
        ".github/workflows/*.yaml",
        ".github/actions/**/*.yml",
        ".github/actions/**/*.yaml",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob.glob(pattern, recursive=True))
    return files


def check_file(path: Path, cache: dict[str, bool]) -> list[str]:
    """Return a list of error strings for *path*."""
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
            errors.append(f"{prefix}\n  → ref '{ref}' is not a 40-character SHA")
            continue

        if not comment or not _VERSION_COMMENT_RE.match(comment):
            errors.append(
                f"{prefix}\n  → missing or invalid version comment"
                f" (expected '# vX.Y.Z', got {comment!r})"
            )
            continue

        if not _verify_sha_tag(action, ref, comment, cache):
            errors.append(
                f"{prefix}\n  → SHA '{ref}' does not match tag '{comment}'"
                f" for {_repo_from_action(action)}"
            )

    return errors


def main() -> int:
    args = sys.argv[1:]
    files = _collect_files(args)
    if not files:
        return 0

    cache = _load_cache()
    all_errors: list[str] = []
    for path in files:
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
