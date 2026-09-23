"""Build shim for branch-2.13.

The mlflow/releases ``python.yml`` release workflow builds each package by running
``python dev/build.py --package-type <type>``. branch-2.13 predates that script and
instead builds via ``dev/build.sh`` (``pyproject.toml`` for ``mlflow``,
``pyproject.skinny.toml`` for ``mlflow-skinny``). This shim provides the workflow's CLI
and delegates to ``dev/build.sh`` so the build matches exactly how 2.13.x shipped.
"""

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an MLflow 2.13 package.")
    parser.add_argument(
        "--package-type",
        required=True,
        choices=["release", "skinny"],
        help="'release' builds the full mlflow package; 'skinny' builds mlflow-skinny.",
    )
    # Accepted for parity with the workflow / v3 build.py interface; unused here.
    parser.add_argument("--sha", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = ["bash", "dev/build.sh"]
    if args.package_type == "skinny":
        cmd.append("--skinny")
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
