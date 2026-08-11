import argparse

from skills.commands import (
    fetch_diff,
    fetch_logs,
    load_rules,
    validate_review,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skills")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_diff.register(subparsers)
    fetch_logs.register(subparsers)
    load_rules.register(subparsers)
    validate_review.register(subparsers)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
