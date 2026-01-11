import argparse

from skills.commands import fetch_diff, fetch_unresolved_comments


def main() -> None:
    parser = argparse.ArgumentParser(prog="skills")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_diff.register(subparsers)
    fetch_unresolved_comments.register(subparsers)

    args = parser.parse_args()
    args.func(args)
