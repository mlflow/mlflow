import argparse

from skills.commands import analyze_ci, fetch_diff, fetch_unresolved_comments, validate_review


def main() -> None:
    parser = argparse.ArgumentParser(prog="skills")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_ci.register(subparsers)
    fetch_diff.register(subparsers)
    fetch_unresolved_comments.register(subparsers)
    validate_review.register(subparsers)

    args = parser.parse_args()
    args.func(args)
