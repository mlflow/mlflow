from __future__ import annotations

import re

from mlflow.exceptions import MlflowException

# Official SemVer 2.0.0 regex (https://semver.org).
_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

# A "semverish" numeric core: 1, 1.2, or 1.2.3 (each component without leading zeros).
_CORE_RE = re.compile(r"^(?:0|[1-9]\d*)(?:\.(?:0|[1-9]\d*)){0,2}$")


def normalize_semver(version: str) -> str:
    """Coerce a semverish version to full SemVer, validate, and return it.

    ``1`` and ``1.2`` are padded to ``1.0.0`` / ``1.2.0``. Prerelease and build
    metadata are preserved. Non-SemVer input raises ``MlflowException``.
    """
    if not isinstance(version, str) or not version.strip():
        raise MlflowException.invalid_parameter_value(
            f"Invalid version {version!r}: version must be a non-empty string."
        )
    version = version.strip()

    # Separate the numeric core from any prerelease ('-') or build metadata ('+').
    if match := re.search(r"[-+]", version):
        core = version[: match.start()]
        suffix = version[match.start() :]
    else:
        core = version
        suffix = ""

    if _CORE_RE.fullmatch(core):
        parts = core.split(".")
        parts += ["0"] * (3 - len(parts))
        core = ".".join(parts)

    candidate = core + suffix
    if _SEMVER_RE.fullmatch(candidate) is None:
        raise MlflowException.invalid_parameter_value(
            f"Invalid SemVer version {version!r}. Expected a version like '1.2.3', "
            "'1.0.0-beta.1', or '1.0.0+build.5'."
        )
    return candidate
