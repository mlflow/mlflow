import argparse

from skills.commands import (
    dedupe_advisories,
    fetch_diff,
    fetch_logs,
    get_advisory,
    list_advisories,
    load_rules,
    validate_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="skills")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dedupe_advisories.register(subparsers)
    fetch_diff.register(subparsers)
    fetch_logs.register(subparsers)
    get_advisory.register(subparsers)
    list_advisories.register(subparsers)
    load_rules.register(subparsers)
    validate_review.register(subparsers)

    args = parser.parse_args()
    args.func(args)
