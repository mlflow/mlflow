# ruff: noqa: T201
"""Print contents of .claude/rules/*.md files whose path globs match changed files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

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
    if not text.startswith("---\n"):
        return []
    end = text.find("\n---", 4)
    if end == -1:
        return []
    front = yaml.safe_load(text[4:end])
    if not isinstance(front, dict):
        return []
    match front.get("paths"):
        case str() as scalar:
            return [scalar]
        case list() as items:
            return [v for v in items if isinstance(v, str)]
        case _:
            return []


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
        "print-rules",
        help=(
            "Read newline-separated file paths from stdin and print the contents of "
            "`.claude/rules/*.md` files whose `paths` glob matches at least one path"
        ),
    ).set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    changed = [s for line in sys.stdin if (s := line.strip())]
    if not changed:
        return
    for rule in matching_rules(changed):
        rel = rule.relative_to(REPO_ROOT)
        body = rule.read_text()
        print(f"================ {rel} ================")
        print(body if body.endswith("\n") else body + "\n")
