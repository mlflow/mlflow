from __future__ import annotations

import argparse
import asyncio

from flavors import _lint, _matrix, _update


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="flavors",
        description="CLI for cooking mlflow/ml-package-versions.yml.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix_parser = subparsers.add_parser(
        "matrix",
        help="Generate the cross-version test matrix.",
        description=_matrix.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _matrix.add_arguments(matrix_parser)
    matrix_parser.set_defaults(func=lambda a: asyncio.run(_matrix.run(a)))

    update_parser = subparsers.add_parser(
        "update",
        help="Update maximum versions in ml-package-versions.yml.",
        description=_update.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _update.add_arguments(update_parser)
    update_parser.set_defaults(func=_update.run)

    lint_parser = subparsers.add_parser(
        "lint",
        help="Lint ml-package-versions.yml.",
        description=_lint.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _lint.add_arguments(lint_parser)
    lint_parser.set_defaults(func=_lint.run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
