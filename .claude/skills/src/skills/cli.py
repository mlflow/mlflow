import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="skills")
    subparsers = parser.add_subparsers(dest="command", required=True)  # noqa: F841

    # Subcommands will be registered here
    # e.g., fetch_diff.register(subparsers)

    args = parser.parse_args()
    args.func(args)
