"""
Utilities for UV package manager integration.

This module provides functions for detecting UV projects and exporting dependencies
via `uv export` for automatic dependency inference during model logging.
"""

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from packaging.version import Version

_logger = logging.getLogger(__name__)

# Minimum UV version required for `uv export` functionality
_MIN_UV_VERSION = Version("0.5.0")

# File names used for UV project detection
_UV_LOCK_FILE = "uv.lock"
_PYPROJECT_FILE = "pyproject.toml"


def get_uv_version() -> Version | None:
    """
    Get the installed UV version.

    Returns:
        The UV version as a packaging.version.Version object, or None if UV is not installed
        or version cannot be determined.
    """
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        return None

    try:
        result = subprocess.run(
            [uv_bin, "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Output format: "uv 0.5.0 (abc123 2024-01-01)"
        version_str = result.stdout.strip().split()[1]
        return Version(version_str)
    except (subprocess.CalledProcessError, IndexError, ValueError) as e:
        _logger.debug(f"Failed to determine UV version: {e}")
        return None


def is_uv_available() -> bool:
    """
    Check if UV is installed and meets the minimum version requirement.

    Returns:
        True if UV is installed and version >= 0.5.0, False otherwise.
    """
    version = get_uv_version()
    if version is None:
        return False

    if version < _MIN_UV_VERSION:
        _logger.debug(f"UV version {version} is below minimum required version {_MIN_UV_VERSION}")
        return False

    return True


def detect_uv_project(directory: str | Path | None = None) -> dict[str, Path] | None:
    """
    Detect if the given directory is a UV project.

    A UV project is identified by the presence of BOTH `uv.lock` and `pyproject.toml`
    in the specified directory.

    Args:
        directory: The directory to check. Defaults to the current working directory.

    Returns:
        A dictionary containing paths to detected files:
        - "uv_lock": Path to uv.lock
        - "pyproject": Path to pyproject.toml
        Returns None if the directory is not a UV project.
    """
    directory = Path.cwd() if directory is None else Path(directory)

    uv_lock_path = directory / _UV_LOCK_FILE
    pyproject_path = directory / _PYPROJECT_FILE

    if uv_lock_path.exists() and pyproject_path.exists():
        _logger.info(
            f"Detected UV project: found {_UV_LOCK_FILE} and {_PYPROJECT_FILE} in {directory}"
        )
        return {
            "uv_lock": uv_lock_path,
            "pyproject": pyproject_path,
        }

    return None


def _evaluate_marker(marker: str, version_info: Any) -> bool:
    """
    Evaluate a PEP 508 environment marker against the current Python version.

    This is a simplified evaluator that handles common markers like:
    - python_version < '3.11'
    - python_full_version >= '3.11'
    - platform_python_implementation != 'PyPy'

    Args:
        marker: The marker string (e.g., "python_version < '3.11'")
        version_info: The Python version info (sys.version_info)

    Returns:
        True if the marker matches the current environment, False otherwise.
    """
    # Extract the current Python version
    py_version = f"{version_info.major}.{version_info.minor}"
    py_full_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    # Normalize the marker
    marker = marker.strip()

    # Handle 'and' conditions by checking all parts
    if " and " in marker:
        parts = marker.split(" and ")
        return all(_evaluate_marker(part.strip(), version_info) for part in parts)

    # Handle 'or' conditions
    if " or " in marker:
        parts = marker.split(" or ")
        return any(_evaluate_marker(part.strip(), version_info) for part in parts)

    # Match pattern: variable operator 'value'
    if not (match := re.match(r"(\w+)\s*(==|!=|<=|>=|<|>)\s*['\"]([^'\"]+)['\"]", marker)):
        # If we can't parse the marker, assume it matches (conservative)
        return True

    var_name, operator, value = match.groups()

    # Get the actual value for comparison
    if var_name in ("python_version", "python_full_version"):
        actual = py_full_version if var_name == "python_full_version" else py_version
    elif var_name == "platform_python_implementation":
        actual = platform.python_implementation()
    elif var_name == "sys_platform":
        actual = sys.platform
    elif var_name == "platform_system":
        actual = platform.system()
    elif var_name == "platform_machine":
        actual = platform.machine()
    elif var_name == "os_name":
        actual = os.name
    else:
        # Unknown marker variable, assume it matches
        return True

    # Compare versions
    if var_name in ("python_version", "python_full_version"):
        actual_v = Version(actual)
        value_v = Version(value)

        match operator:
            case "==":
                return actual_v == value_v
            case "!=":
                return actual_v != value_v
            case "<":
                return actual_v < value_v
            case "<=":
                return actual_v <= value_v
            case ">":
                return actual_v > value_v
            case ">=":
                return actual_v >= value_v
    else:
        # String comparison for platform_python_implementation, etc.
        match operator:
            case "==":
                return actual == value
            case "!=":
                return actual != value
            case _:
                return True

    return True


def export_uv_requirements(
    directory: str | Path | None = None,
    no_dev: bool = True,
    no_hashes: bool = True,
    frozen: bool = True,
    uv_lock: str | Path | None = None,
) -> list[str] | None:
    """
    Export dependencies from a UV project to pip-compatible requirements.

    Runs `uv export` to generate a list of pinned dependencies from the UV lock file.

    Args:
        directory: The UV project directory. Defaults to the current working directory.
            Ignored if uv_lock is provided.
        no_dev: Exclude development dependencies. Defaults to True.
        no_hashes: Omit hashes from output. Defaults to True.
        frozen: Use frozen lockfile without updating. Defaults to True.
        uv_lock: Explicit path to uv.lock file. When provided, the UV project directory
            is derived from this path (parent directory). Useful for monorepos.

    Returns:
        A list of requirement strings (e.g., ["requests==2.28.0", "numpy==1.24.0"]),
        or None if export fails.
    """
    if not is_uv_available():
        _logger.warning(
            "UV is not available or version is below minimum required. "
            "Falling back to pip-based inference."
        )
        return None

    uv_bin = shutil.which("uv")

    # If explicit uv_lock path provided, derive directory from it
    if uv_lock is not None:
        uv_lock_path = Path(uv_lock)
        if not uv_lock_path.exists():
            _logger.warning(f"Specified uv_lock path does not exist: {uv_lock_path}")
            return None
        directory = uv_lock_path.parent
        _logger.info(f"Using explicit uv_lock path for export: {uv_lock_path}")
    else:
        directory = Path.cwd() if directory is None else Path(directory)

    cmd = [uv_bin, "export"]

    if no_dev:
        cmd.append("--no-dev")
    if no_hashes:
        cmd.append("--no-hashes")
    if frozen:
        cmd.append("--frozen")

    # Additional flags for cleaner output
    cmd.extend(
        [
            "--no-header",  # Omit the "autogenerated by uv" comment
            "--no-emit-project",  # Don't emit the project itself
        ]
    )

    try:
        _logger.debug(f"Running UV export: {' '.join(str(c) for c in cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=directory,
        )

        # Parse output into list of requirements
        # UV may output multiple entries for the same package with different environment
        # markers (e.g., numpy==2.2.6 ; python_version < '3.11' vs numpy==2.4.1).
        # We filter to keep only requirements matching the current Python version.
        requirements = []
        seen_packages = set()

        for line in result.stdout.strip().splitlines():
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Handle indented "# via" comments that uv may include
            if line.startswith(" "):
                continue

            # Check for environment markers (separated by ';')
            if ";" in line:
                req_part, marker_part = line.split(";", 1)
                req_part = req_part.strip()
                marker_part = marker_part.strip()

                # Evaluate marker against current Python version
                if not _evaluate_marker(marker_part, sys.version_info):
                    continue
                line = req_part  # Use requirement without marker

            # Extract package name (before ==, >=, <=, <, >, [, etc.)
            pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0]
            pkg_name = pkg_name.split("<")[0].split(">")[0].split("[")[0].strip().lower()

            # Skip if we've already seen this package
            if pkg_name in seen_packages:
                continue
            seen_packages.add(pkg_name)
            requirements.append(line)

        _logger.info(f"Exported {len(requirements)} dependencies via UV")
        return requirements

    except subprocess.CalledProcessError as e:
        _logger.warning(
            f"UV export failed with exit code {e.returncode}. "
            f"stderr: {e.stderr}. Falling back to pip-based inference."
        )
        return None
    except Exception as e:
        _logger.warning(f"UV export failed: {e}. Falling back to pip-based inference.")
        return None


def get_python_version_from_uv_project(
    directory: str | Path | None = None,
    uv_lock: str | Path | None = None,
) -> str | None:
    """
    Extract Python version from a UV project.

    Checks for `.python-version` file first, then falls back to parsing
    `requires-python` from `pyproject.toml`.

    Args:
        directory: The UV project directory. Defaults to the current working directory.
            Ignored if uv_lock is provided.
        uv_lock: Explicit path to uv.lock file. When provided, the UV project directory
            is derived from this path (parent directory).

    Returns:
        Python version string (e.g., "3.11.5" or "3.11"), or None if not found.
    """
    # If explicit uv_lock path provided, derive directory from it
    if uv_lock is not None:
        uv_lock_path = Path(uv_lock)
        if uv_lock_path.exists():
            directory = uv_lock_path.parent
        else:
            return None
    else:
        directory = Path.cwd() if directory is None else Path(directory)

    # Check .python-version file first
    python_version_file = directory / ".python-version"
    if python_version_file.exists():
        if version := python_version_file.read_text().strip():
            _logger.debug(f"Found Python version {version} from .python-version")
            return version

    # Fall back to pyproject.toml requires-python (simple regex parsing to avoid tomli dep)
    pyproject_path = directory / _PYPROJECT_FILE
    if pyproject_path.exists():
        try:
            content = pyproject_path.read_text()
            # Match requires-python = ">=3.10" or requires-python = "3.11"
            if match := re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', content):
                requires_python = match.group(1)
                # Extract version from specifier like ">=3.10" -> "3.10"
                if version_match := re.search(r"(\d+\.\d+(?:\.\d+)?)", requires_python):
                    version = version_match.group(1)
                    _logger.debug(
                        f"Found Python version {version} from pyproject.toml requires-python"
                    )
                    return version
        except Exception as e:
            _logger.debug(f"Failed to parse pyproject.toml for Python version: {e}")

    return None


# File names for UV artifacts
_UV_LOCK_ARTIFACT_NAME = "uv.lock"
_PYPROJECT_ARTIFACT_NAME = "pyproject.toml"
_PYTHON_VERSION_FILE = ".python-version"

# Environment variable to disable UV file logging (for large projects)
_MLFLOW_LOG_UV_FILES_ENV = "MLFLOW_LOG_UV_FILES"


def _should_log_uv_files() -> bool:
    """Check if UV files should be logged based on environment variable."""
    env_value = os.environ.get(_MLFLOW_LOG_UV_FILES_ENV, "true").lower()
    return env_value not in ("false", "0", "no")


def copy_uv_project_files(
    dest_dir: str | Path,
    source_dir: str | Path | None = None,
    uv_lock: str | Path | None = None,
) -> bool:
    """
    Copy UV project files to the model artifact directory.

    Copies uv.lock, pyproject.toml, and .python-version (if present) to preserve
    UV project configuration as model artifacts for reproducibility.

    Can be disabled by setting MLFLOW_LOG_UV_FILES=false environment variable
    for large projects where uv.lock size is a concern.

    Args:
        dest_dir: The destination directory (model artifact directory).
        source_dir: The source directory containing UV project files.
            Defaults to the current working directory. Ignored if uv_lock is provided.
        uv_lock: Explicit path to uv.lock file. When provided, the UV project directory
            is derived from this path (parent directory). Useful for monorepos or
            non-standard project layouts where uv.lock is not in CWD.

    Returns:
        True if UV files were copied, False otherwise.
    """
    # Check if UV file logging is disabled via environment variable
    if not _should_log_uv_files():
        _logger.info(
            f"UV file logging disabled via {_MLFLOW_LOG_UV_FILES_ENV} environment variable"
        )
        return False

    dest_dir = Path(dest_dir)

    # If explicit uv_lock path provided, derive source_dir from it
    if uv_lock is not None:
        uv_lock_path = Path(uv_lock)
        if not uv_lock_path.exists():
            _logger.warning(f"Specified uv_lock path does not exist: {uv_lock_path}")
            return False
        source_dir = uv_lock_path.parent
        _logger.info(f"Using explicit uv_lock path: {uv_lock_path}")
    else:
        source_dir = Path.cwd() if source_dir is None else Path(source_dir)

    uv_project = detect_uv_project(source_dir)

    if uv_project is None:
        if uv_lock is not None:
            _logger.warning(
                f"Explicit uv_lock provided but pyproject.toml not found in {source_dir}"
            )
        return False

    uv_lock_src = uv_project["uv_lock"]
    pyproject_src = uv_project["pyproject"]
    python_version_src = source_dir / _PYTHON_VERSION_FILE

    try:
        # Copy uv.lock
        uv_lock_dest = dest_dir / _UV_LOCK_ARTIFACT_NAME
        shutil.copy2(uv_lock_src, uv_lock_dest)
        _logger.info(f"Copied {_UV_LOCK_ARTIFACT_NAME} to model artifacts")

        # Copy pyproject.toml
        pyproject_dest = dest_dir / _PYPROJECT_ARTIFACT_NAME
        shutil.copy2(pyproject_src, pyproject_dest)
        _logger.info(f"Copied {_PYPROJECT_ARTIFACT_NAME} to model artifacts")

        # Copy .python-version if it exists
        if python_version_src.exists():
            python_version_dest = dest_dir / _PYTHON_VERSION_FILE
            shutil.copy2(python_version_src, python_version_dest)
            _logger.info(f"Copied {_PYTHON_VERSION_FILE} to model artifacts")

        return True
    except Exception as e:
        _logger.warning(f"Failed to copy UV project files: {e}")
        return False
