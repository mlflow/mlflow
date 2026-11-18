"""Validate a pull request review payload in two stages.

1. Shape: check the payload against a JSON Schema (review.schema.yml unless --schema is given).
2. Positions: with --diff, check that every comment points at a row of the diff.

Usage:
    uv run --no-project .claude/scripts/validate_review.py --payload payload.json --diff pr.diff
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

_SCHEMA = Path(__file__).with_name("review.schema.yml")
_HUNK_HEADER = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_Side = Literal["LEFT", "RIGHT"]
_Anchor = tuple[_Side, int]
# Maps each anchor in a hunk to its row, so a range can be checked for order.
_Hunk = dict[_Anchor, int]
_ROW_SIDES: dict[str, tuple[_Side, ...]] = {
    "-": ("LEFT",),
    "+": ("RIGHT",),
    " ": ("LEFT", "RIGHT"),
}


def _read_path(header: str, prefix: str) -> str | None:
    path = header[4:].split("\t", 1)[0]
    if path == "/dev/null":
        return None
    if path.startswith('"') and path.endswith('"'):
        # Git uses C-style escapes, with one octal escape per UTF-8 byte.
        path = ast.literal_eval(path).encode("latin-1").decode()
    return path.removeprefix(prefix)


def _read_hunk(header: str, lines: Iterator[str]) -> _Hunk:
    if not (match := _HUNK_HEADER.match(header)):
        raise ValueError(f"Invalid hunk header: {header!r}")

    old_start, old_count, new_start, new_count = match.groups()
    current: dict[_Side, int] = {"LEFT": int(old_start), "RIGHT": int(new_start)}
    stops: dict[_Side, int] = {
        "LEFT": current["LEFT"] + int(old_count or 1),
        "RIGHT": current["RIGHT"] + int(new_count or 1),
    }
    hunk: _Hunk = {}
    row = 0
    # The header's line counts, not the next header, decide where the hunk ends.
    while current != stops:
        line = next(lines, None)
        if line is None:
            raise ValueError(f"Diff ends inside hunk {header!r}")
        if line.startswith("\\"):
            continue
        # An empty line is an unchanged blank line under `diff.suppressBlankEmpty`.
        sides = _ROW_SIDES.get(line[:1] or " ", ())
        if not sides or any(current[s] >= stops[s] for s in sides):
            raise ValueError(f"Hunk body does not match {header!r}")
        for side in sides:
            hunk[side, current[side]] = row
            current[side] += 1
        row += 1
    return hunk


def read_hunks(diff: str) -> dict[str, list[_Hunk]]:
    files: dict[str, list[_Hunk]] = {}
    old_path = path = None
    # Drop the final newline so it is not read as a blank row of a truncated hunk.
    lines = iter(diff.removesuffix("\n").split("\n"))
    for line in lines:
        if line.startswith("diff --git "):
            old_path = path = None
        elif line.startswith("--- "):
            old_path = _read_path(line, "a/")
        elif line.startswith("+++ "):
            path = _read_path(line, "b/") or old_path
        elif line.startswith("@@"):
            if path is None:
                raise ValueError(f"Hunk {line!r} is missing a file path")
            files.setdefault(path, []).append(_read_hunk(line, lines))
    return files


def _label(anchor: _Anchor) -> str:
    side, line = anchor
    return f"{side[0]}{line}"


def _check_comment(comment: dict[str, Any], files: dict[str, list[_Hunk]]) -> str | None:
    path = comment["path"]
    if not (hunks := files.get(path)):
        return f"{path} has no lines in the diff"

    end: _Anchor = (comment["side"], comment["line"])
    anchors = [end]
    if "start_line" in comment:
        anchors.insert(0, (comment["start_side"], comment["start_line"]))

    found = [next((h for h in hunks if a in h), None) for a in anchors]
    if missing := [_label(a) for a, h in zip(anchors, found) if h is None]:
        return f"{' and '.join(missing)} not in the diff of {path}"

    match anchors, [h for h in found if h is not None]:
        case [start, end], [start_hunk, end_hunk]:
            if start_hunk is not end_hunk:
                return f"{_label(start)} and {_label(end)} are in different hunks of {path}"
            if start_hunk[start] >= start_hunk[end]:
                return f"{_label(start)} does not come before {_label(end)} in {path}"
    return None


def check_positions(payload: dict[str, Any], diff: str) -> list[str]:
    files = read_hunks(diff)
    return [
        f"comments[{i}]: {error}"
        for i, c in enumerate(payload["comments"])
        if (error := _check_comment(c, files))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a review payload's shape and its comment positions in a Git diff."
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=_SCHEMA,
        help="JSON Schema for the payload (default: %(default)s). With --diff, it must "
        "require `comments` and each comment's `path`, `side` and `line`.",
    )
    parser.add_argument("--payload", type=Path, required=True, help="Review payload JSON file")
    parser.add_argument("--diff", type=Path, help="Git diff the comments must point into")
    args = parser.parse_args()

    # Positions are only meaningful for a well-formed payload, so stop at the first stage.
    schema_check = subprocess.run(
        ["uvx", "check-jsonschema@0.37.4", "--schemafile", args.schema, args.payload],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if schema_check.returncode:
        raise SystemExit(schema_check.stdout.strip())

    if args.diff is None:
        return
    payload = json.loads(args.payload.read_text())
    try:
        errors = check_positions(payload, args.diff.read_text())
    except ValueError as e:
        raise SystemExit(f"Invalid diff: {e}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
