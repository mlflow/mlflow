import argparse

from skills.commands import (
    annotate_diff,
    embed_media,
    fetch_logs,
    load_rules,
    upload_media,
    validate_review,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skills")
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate_diff.register(subparsers)
    embed_media.register(subparsers)
    fetch_logs.register(subparsers)
    load_rules.register(subparsers)
    upload_media.register(subparsers)
    validate_review.register(subparsers)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
