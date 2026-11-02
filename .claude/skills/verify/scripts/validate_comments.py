import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

HUNK_HEADER = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
Side = Literal["LEFT", "RIGHT"]


def diff_path(header: str) -> str:
    path = header[4:].split("\t", 1)[0]
    if path.startswith('"'):
        path = ast.literal_eval(path)
    return path[2:]


def diff_hunks(diff: str) -> dict[str, list[dict[tuple[Side, int], int]]]:
    files: dict[str, list[dict[tuple[Side, int], int]]] = {}
    path = ""
    hunk: dict[tuple[Side, int], int] | None = None
    old = new = 0
    for position, line in enumerate(diff.splitlines()):
        if line.startswith("diff --git "):
            path = ""
            hunk = None
        elif hunk is None and line.startswith(("--- ", "+++ ")):
            if line[4:].split("\t", 1)[0] != "/dev/null":
                path = diff_path(line)
        elif match := HUNK_HEADER.match(line):
            old, new = map(int, match.groups())
            hunk = {}
            files.setdefault(path, []).append(hunk)
        elif hunk is not None:
            if line.startswith((" ", "-")):
                hunk["LEFT", old] = position
                old += 1
            if line.startswith((" ", "+")):
                hunk["RIGHT", new] = position
                new += 1
    return files


def validate_comments(payload: dict[str, Any], diff: str) -> list[str]:
    files = diff_hunks(diff)
    errors = []
    for index, comment in enumerate(payload["comments"], start=1):
        positions = [(comment["side"], comment["line"])]
        if "start_line" in comment:
            positions.insert(0, (comment["start_side"], comment["start_line"]))
        if not any(
            all(position in hunk for position in positions)
            and (len(positions) == 1 or hunk[positions[0]] < hunk[positions[-1]])
            for hunk in files.get(comment["path"], [])
        ):
            errors.append(f"Comment {index}: position or ordered range is not within one diff hunk")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate inline comments against a diff on stdin")
    parser.add_argument(
        "--payload", type=Path, required=True, help="Path to the review payload JSON"
    )
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text())
    if errors := validate_comments(payload, sys.stdin.read()):
        raise SystemExit("\n".join(errors))
    print("Inline comments are valid.")  # noqa: T201


if __name__ == "__main__":
    main()
