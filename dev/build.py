"""Legacy build shim for branch-2.11.

The mlflow/releases ``python.yml`` release workflow builds each package by running
``python dev/build.py --package-type <type>``. That script only exists on v3+
branches, which use a multi-package pyproject layout. branch-2.11 predates it and
uses the legacy single-``setup.py`` layout: the package name is toggled by the
``MLFLOW_SKINNY`` env var and the version comes from ``mlflow/version.py``.

This shim provides the same CLI so the unmodified workflow can build 2.11:
  - release: the full ``mlflow`` package (bundles the pre-built JS UI in
    ``mlflow/server/js/build``, which the workflow builds before calling this)
  - skinny:  ``mlflow-skinny`` (``MLFLOW_SKINNY=1``; no UI)

The wheel is built directly from the source tree (not from the sdist) so that
setup.py ``package_data`` is bundled -- the JS UI for release, and the alembic /
recipes data for skinny. A plain ``python -m build`` (sdist -> wheel) would drop
those, because ``MANIFEST.in`` does not ship them in the sdist.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

JS_BUILD_DIR = Path("mlflow/server/js/build")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an MLflow 2.11 package.")
    parser.add_argument(
        "--package-type",
        required=True,
        choices=["release", "skinny"],
        help="'release' builds the full mlflow package; 'skinny' builds mlflow-skinny.",
    )
    # Accepted for parity with the workflow / v3 build.py interface; unused here.
    parser.add_argument("--sha", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def build(extra_env: dict) -> None:
    env = {**os.environ, **extra_env}
    # Wheel first, directly from the tree, so setup.py package_data is bundled; then the sdist.
    subprocess.check_call([sys.executable, "-m", "build", "--wheel"], env=env)
    subprocess.check_call([sys.executable, "-m", "build", "--sdist"], env=env)


def main() -> None:
    args = parse_args()
    if args.package_type == "release":
        if not JS_BUILD_DIR.exists() or not any(JS_BUILD_DIR.iterdir()):
            sys.exit(
                f"ERROR: {JS_BUILD_DIR} is missing or empty. Build the UI "
                "(cd mlflow/server/js && yarn install && yarn build) before the release package."
            )
        # Clear MLFLOW_SKINNY so an ambient value can't make setup.py build skinny here
        # ("" is falsey in setup.py's `bool(os.environ.get("MLFLOW_SKINNY"))` check).
        build({"MLFLOW_SKINNY": ""})
    else:  # skinny
        build({"MLFLOW_SKINNY": "1"})


if __name__ == "__main__":
    main()
