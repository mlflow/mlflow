"""Validate that all remote GitHub Actions are SHA-pinned with a version comment."""

import json
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
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

# Requires at least vMAJOR.MINOR.PATCH to avoid ambiguous moving tags like v4.
_VERSION_COMMENT_RE = re.compile(r"^v\d+\.\d+\.\d+(?:\.\d+)*$")

_CACHE_PATH = Path(".cache/action-pins.json")


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
    try:
        result = _resolve_tag(repo, sha, tag)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    cache[cache_key] = result
    return result


def _resolve_tag(repo: str, sha: str, tag: str) -> bool:
    output = subprocess.check_output(
        ["git", "ls-remote", "--tags", f"https://github.com/{repo}.git", tag],
        text=True,
        timeout=10,
    )
    return any(line.split()[0] == sha for line in output.splitlines() if line)


def _iter_files() -> Iterator[Path]:
    root = Path(".github")
    for pattern in (
        "workflows/*.yml",
        "workflows/*.yaml",
        "actions/**/*.yml",
        "actions/**/*.yaml",
    ):
        yield from root.glob(pattern)


@dataclass(frozen=True, slots=True)
class ActionRef:
    prefix: str
    action: str
    ref: str
    comment: str | None


def _iter_actions(path: Path) -> Iterator[ActionRef]:
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if m := _USES_RE.match(line):
                action = m.group("action")
                if not action.startswith("./"):
                    prefix = f"{path}:{lineno}: {line.strip()!r}"
                    yield ActionRef(prefix, action, m.group("ref"), m.group("comment"))


def _check_action(a: ActionRef, cache: dict[str, bool]) -> str | None:
    if not _SHA_RE.match(a.ref):
        return f"{a.prefix}\n  error: ref '{a.ref}' is not a 40-character SHA"

    if not a.comment or not _VERSION_COMMENT_RE.match(a.comment):
        return (
            f"{a.prefix}\n  error: missing or invalid version comment"
            f" (expected '# vX.Y.Z', got {a.comment!r})"
        )

    verified = _verify_sha_tag(a.action, a.ref, a.comment, cache)
    if verified is None:
        return (
            f"{a.prefix}\n  error: could not verify SHA against tag '{a.comment}'"
            f" for {_repo_from_action(a.action)} (GitHub API unavailable)"
        )
    if not verified:
        return (
            f"{a.prefix}\n  error: SHA '{a.ref}' does not match tag '{a.comment}'"
            f" for {_repo_from_action(a.action)}"
        )
    return None


def _check_version_consistency(all_action_refs: list[ActionRef]) -> Iterator[str]:
    by_action: dict[str, list[ActionRef]] = defaultdict(list)
    for action_ref in all_action_refs:
        by_action[action_ref.action].append(action_ref)

    for action, refs in sorted(by_action.items()):
        versions = {(ref.ref, ref.comment) for ref in refs}
        if len(versions) > 1:
            lines = "\n".join(f"  {ref.prefix}" for ref in sorted(refs, key=lambda r: r.prefix))
            yield f"{action} is pinned to multiple versions:\n{lines}"


def main() -> int:
    cache = _load_cache()
    all_errors: list[str] = []
    all_action_refs: list[ActionRef] = []
    try:
        for path in _iter_files():
            for action_ref in _iter_actions(path):
                if error := _check_action(action_ref, cache):
                    all_errors.append(error)
                else:
                    all_action_refs.append(action_ref)
    finally:
        _save_cache(cache)
    all_errors.extend(_check_version_consistency(all_action_refs))

    if all_errors:
        print("action-pins: the following violations were found:\n", file=sys.stderr)
        for err in all_errors:
            print(err, file=sys.stderr)
        print(f"\n{len(all_errors)} violation(s) found.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
