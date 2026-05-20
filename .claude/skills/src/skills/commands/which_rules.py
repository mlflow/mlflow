# ruff: noqa: T201
"""Print .claude/rules/*.md files whose path globs match a set of changed files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RULES_DIR = Path(__file__).parents[4] / "rules"
REPO_ROOT = Path(__file__).parents[5]


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern[i : i + 3] == "**/":
            parts.append("(?:.*/)?")
            i += 3
        elif pattern[i : i + 2] == "**":
            parts.append(".*")
            i += 2
        elif pattern[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(pattern[i]))
            i += 1
    return re.compile(r"\A" + "".join(parts) + r"\Z")


def parse_frontmatter_paths(text: str) -> list[str]:
    """Extract ``paths`` values from a leading YAML frontmatter block.

    Supports a single quoted/unquoted scalar or a YAML list. Returns [] when
    there is no frontmatter or no ``paths`` key.
    """
    if not text.startswith("---\n"):
        return []
    end = text.find("\n---", 4)
    if end == -1:
        return []
    patterns: list[str] = []
    in_list = False
    for line in text[4:end].splitlines():
        if in_list:
            if m := re.match(r"\s*-\s*(.+?)\s*$", line):
                patterns.append(m.group(1).strip("\"'"))
                continue
            in_list = False
        if m := re.match(r"paths:\s*(.*)$", line):
            if value := m.group(1).strip():
                patterns.append(value.strip("\"'"))
            else:
                in_list = True
    return patterns


def matching_rules(changed: list[str], rules_dir: Path = RULES_DIR) -> list[Path]:
    matched: list[Path] = []
    for rule in sorted(rules_dir.glob("*.md")):
        patterns = parse_frontmatter_paths(rule.read_text())
        if not patterns:
            continue
        regexes = [glob_to_regex(p) for p in patterns]
        if any(r.match(p) for r in regexes for p in changed):
            matched.append(rule)
    return matched


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    subparsers.add_parser(
        "which-rules",
        help=(
            "Read newline-separated file paths from stdin and print .claude/rules/*.md "
            "files whose `paths` glob matches at least one path"
        ),
    ).set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    changed = [line.strip() for line in sys.stdin if line.strip()]
    if not changed:
        return
    for rule in matching_rules(changed):
        print(rule.relative_to(REPO_ROOT))
