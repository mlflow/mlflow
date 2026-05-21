"""
Backward-compatibility wrapper around `flavors update`.

Prefer `uv run flavors update`. This script exists for internal jobs that
still invoke `python dev/update_ml_package_versions.py`; it will be removed
once they migrate.

TODO: Delete this file once all internal jobs have migrated to
`flavors update`.
"""

import argparse
import sys

# Re-exported for legacy importers (tests, internal scripts).
from flavors._update import (  # noqa: F401
    VersionInfo,
    check_pypi_accessibility,
    get_min_supported_version,
    get_package_version_infos,
    get_packages,
    update,
    update_ml_package_versions_py,
    update_version,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update MLflow package versions")
    parser.add_argument(
        "--skip-yml", action="store_true", help="Skip updating ml-package-versions.yml"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    update(_parse_args(sys.argv[1:]).skip_yml)
