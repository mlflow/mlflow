# ruff: noqa: T201
"""Validate a pr-review JSON payload against the schema in ``.claude/skills/pr-review/``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

DEFAULT_SCHEMA = Path(__file__).parents[3] / "pr-review" / "review-payload.schema.json"


def format_path(path: list[str | int]) -> str:
    if not path:
        return "<root>"
    out = ""
    for p in path:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            out += f".{p}" if out else str(p)
    return out


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "validate-review",
        help="Validate a pr-review payload against the JSON schema",
    )
    parser.add_argument("payload", type=Path, help="Path to the review payload JSON file")
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help=f"Path to the JSON schema (default: {DEFAULT_SCHEMA})",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    schema = json.loads(args.schema.read_text())
    payload = json.loads(args.payload.read_text())

    validator = Draft202012Validator(schema)
    if errors := sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path)):
        print(f"ERROR: {args.payload} failed schema validation", file=sys.stderr)
        for err in errors:
            print(f"  {format_path(list(err.absolute_path))}: {err.message}", file=sys.stderr)
        sys.exit(1)

    n = len(payload.get("comments", []))
    print(f"OK: event={payload['event']}, comments={n}")
