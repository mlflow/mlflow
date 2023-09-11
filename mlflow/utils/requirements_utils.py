"""
This module provides a set of utilities for interpreting and creating requirements files
(e.g. pip's `requirements.txt`), which is useful for managing ML software environments.
"""

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import namedtuple
from itertools import chain, filterfalse
from pathlib import Path
from threading import Timer
from typing import NamedTuple, Optional

import importlib_metadata
import pkg_resources  # noqa: TID251
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

import mlflow
from mlflow.environment_variables import MLFLOW_REQUIREMENTS_INFERENCE_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils.versioning import _strip_dev_version_suffix
from mlflow.utils.databricks_utils import is_in_databricks_runtime

_logger = logging.getLogger(__name__)


def _is_comment(line):
    return line.startswith("#")


def _is_empty(line):
    return line == ""


def _strip_inline_comment(line):
    return line[: line.find(" #")].rstrip() if " #" in line else line


def _is_requirements_file(line):
    return line.startswith("-r ") or line.startswith("--requirement ")


def _is_constraints_file(line):
    return line.startswith("-c ") or line.startswith("--constraint ")


def _join_continued_lines(lines):
    """
    Joins lines ending with '\\'.

    >>> _join_continued_lines["a\\", "b\\", "c"]
    >>> 'abc'
    """
    continued_lines = []

    for line in lines:
        if line.endswith("\\"):
            continued_lines.append(line.rstrip("\\"))
        else:
            continued_lines.append(line)
            yield "".join(continued_lines)
            continued_lines.clear()

    # The last line ends with '\'
    if continued_lines:
        yield "".join(continued_lines)


# Represents a pip requirement.
#
# :param req_str: A requirement string (e.g. "scikit-learn == 0.24.2").
# :param is_constraint: A boolean indicating whether this requirement is a constraint.
_Requirement = namedtuple("_Requirement", ["req_str", "is_constraint"])


def _parse_requirements(requirements, is_constraint, base_dir=None):
    """
    A simplified version of `pip._internal.req.parse_requirements` which performs the following
    operations on the given requirements file and yields the parsed requirements.

    - Remove comments and blank lines
    - Join continued lines
    - Resolve requirements file references (e.g. '-r requirements.txt')
    - Resolve constraints file references (e.g. '-c constraints.txt')

    :param requirements: A string path to a requirements file on the local filesystem or
                         an iterable of pip requirement strings.
    :param is_constraint: Indicates the parsed requirements file is a constraint file.
    :param base_dir: If specified, resolve relative file references (e.g. '-r requirements.txt')
                     against the specified directory.
    :return: A list of ``_Requirement`` instances.

    References:
    - `pip._internal.req.parse_requirements`:
      https://github.com/pypa/pip/blob/7a77484a492c8f1e1f5ef24eaf71a43df9ea47eb/src/pip/_internal/req/req_file.py#L118
    - Requirements File Format:
      https://pip.pypa.io/en/stable/cli/pip_install/#requirements-file-format
    - Constraints Files:
      https://pip.pypa.io/en/stable/user_guide/#constraints-files
    """
    if base_dir is None:
        if isinstance(requirements, (str, Path)):
            base_dir = os.path.dirname(requirements)
            with open(requirements) as f:
                requirements = f.read().splitlines()
        else:
            base_dir = os.getcwd()

    lines = map(str.strip, requirements)
    lines = map(_strip_inline_comment, lines)
    lines = _join_continued_lines(lines)
    lines = filterfalse(_is_comment, lines)
    lines = filterfalse(_is_empty, lines)

    for line in lines:
        if _is_requirements_file(line):
            req_file = line.split(maxsplit=1)[1]
            # If `req_file` is an absolute path, `os.path.join` returns `req_file`:
            # https://docs.python.org/3/library/os.path.html#os.path.join
            abs_path = os.path.join(base_dir, req_file)
            yield from _parse_requirements(abs_path, is_constraint=False)
        elif _is_constraints_file(line):
            req_file = line.split(maxsplit=1)[1]
            abs_path = os.path.join(base_dir, req_file)
            yield from _parse_requirements(abs_path, is_constraint=True)
        else:
            yield _Requirement(line, is_constraint)


def _flatten(iterable):
    return chain.from_iterable(iterable)


# https://www.python.org/dev/peps/pep-0508/#names
_PACKAGE_NAME_REGEX = re.compile(r"^(\w+|\w+[\w._-]*\w+)")


def _get_package_name(requirement):
    m = _PACKAGE_NAME_REGEX.match(requirement)
    return m and m.group(1)


_NORMALIZE_REGEX = re.compile(r"[-_.]+")


def _normalize_package_name(pkg_name):
    """
    Normalizes a package name using the rule defined in PEP 503:
    https://www.python.org/dev/peps/pep-0503/#normalized-names
    """
    return _NORMALIZE_REGEX.sub("-", pkg_name).lower()


def _get_requires(pkg_name):
    norm_pkg_name = _normalize_package_name(pkg_name)
    if package := pkg_resources.working_set.by_key.get(norm_pkg_name):
        for req in package.requires():
            yield _normalize_package_name(req.name)


def _get_requires_recursive(pkg_name, seen_before=None):
    """
    Recursively yields both direct and transitive dependencies of the specified
    package.
    """
    norm_pkg_name = _normalize_package_name(pkg_name)
    seen_before = seen_before or {norm_pkg_name}
    for req in _get_requires(pkg_name):
        # Prevent infinite recursion due to cyclic dependencies
        if req in seen_before:
            continue
        seen_before.add(req)
        yield req
        yield from _get_requires_recursive(req, seen_before)


def _prune_packages(packages):
    """
    Prunes packages required by other packages. For example, `["scikit-learn", "numpy"]` is pruned
    to `["scikit-learn"]`.
    """
    packages = set(packages)
    requires = set(_flatten(map(_get_requires_recursive, packages)))
    # Do not exclude mlflow's dependencies
    return packages - (requires - set(_get_requires("mlflow")))


def _run_command(cmd, timeout_seconds, env=None):
    """
    Runs the specified command. If it exits with non-zero status, `MlflowException` is raised.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    timer = Timer(timeout_seconds, proc.kill)
    try:
        timer.start()
        stdout, stderr = proc.communicate()
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")
        if proc.returncode != 0:
            msg = "\n".join(
                [
                    f"Encountered an unexpected error while running {cmd}",
                    f"exit status: {proc.returncode}",
                    f"stdout: {stdout}",
                    f"stderr: {stderr}",
                ]
            )
            raise MlflowException(msg)
    finally:
        if timer.is_alive():
            timer.cancel()


def _get_installed_version(package, module=None):
    """
    Obtains the installed package version using `importlib_metadata.version`. If it fails, use
    `__import__(module or package).__version__`.
    """
    try:
        version = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        # Note `importlib_metadata.version(package)` is not necessarily equal to
        # `__import__(package).__version__`. See the example for pytorch below.
        #
        # Example
        # -------
        # $ pip install torch==1.9.0
        # $ python -c "import torch; print(torch.__version__)"
        # 1.9.0+cu102
        # $ python -c "import importlib_metadata; print(importlib_metadata.version('torch'))"
        # 1.9.0
        version = __import__(module or package).__version__

    # Strip the suffix from `dev` versions of PySpark, which are not available for installation
    # from Anaconda or PyPI
    if package == "pyspark":
        version = _strip_dev_version_suffix(version)

    return version


def _capture_imported_modules(model_uri, flavor):
    """
    Runs `_capture_modules.py` in a subprocess and captures modules imported during the model
    loading procedure.
    If flavor is `transformers`, `_capture_transformers_modules.py` is run instead.

    :param model_uri: The URI of the model.
    :param: flavor: The flavor name of the model.
    :return: A list of captured modules.
    """
    local_model_path = _download_artifact_from_uri(model_uri)

    process_timeout = MLFLOW_REQUIREMENTS_INFERENCE_TIMEOUT.get()

    # Run `_capture_modules.py` to capture modules imported during the loading procedure
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "imported_modules.txt")
        # Pass the main environment variables to the subprocess for environment variable mapping
        main_env = os.environ.copy()
        # Reset the path variable from the main process so that the subprocess retains all
        # main process configuration that a user has.
        # See: ``https://github.com/mlflow/mlflow/issues/6905`` for context on minio configuration
        # resolution in a subprocess based on PATH entries.
        main_env["PATH"] = "/usr/sbin:/sbin:" + main_env["PATH"]

        if flavor == mlflow.transformers.FLAVOR_NAME:
            # Lazily import `_capture_transformers_module` here to avoid circular imports.
            from mlflow.utils import _capture_transformers_modules

            for module_to_throw in ["tensorflow", "torch"]:
                try:
                    _run_command(
                        [
                            sys.executable,
                            _capture_transformers_modules.__file__,
                            "--model-path",
                            local_model_path,
                            "--flavor",
                            flavor,
                            "--output-file",
                            output_file,
                            "--sys-path",
                            json.dumps(sys.path),
                            "--module-to-throw",
                            module_to_throw,
                        ],
                        timeout_seconds=process_timeout,
                        env=main_env,
                    )
                    with open(output_file) as f:
                        return f.read().splitlines()

                except MlflowException:
                    pass

        # Lazily import `_capture_module` here to avoid circular imports.
        from mlflow.utils import _capture_modules

        _run_command(
            [
                sys.executable,
                _capture_modules.__file__,
                "--model-path",
                local_model_path,
                "--flavor",
                flavor,
                "--output-file",
                output_file,
                "--sys-path",
                json.dumps(sys.path),
            ],
            timeout_seconds=process_timeout,
            env=main_env,
        )

        with open(output_file) as f:
            return f.read().splitlines()


DATABRICKS_MODULES_TO_PACKAGES = {
    "databricks.automl": ["databricks-automl-runtime"],
    "databricks.automl_runtime": ["databricks-automl-runtime"],
    "databricks.model_monitoring": ["databricks-model-monitoring"],
}
_MODULES_TO_PACKAGES = None
_PACKAGES_TO_MODULES = None


def _init_modules_to_packages_map():
    global _MODULES_TO_PACKAGES
    if _MODULES_TO_PACKAGES is None:
        # Note `importlib_metada.packages_distributions` only captures packages installed into
        # Python’s site-packages directory via tools such as pip:
        # https://importlib-metadata.readthedocs.io/en/latest/using.html#using-importlib-metadata
        _MODULES_TO_PACKAGES = importlib_metadata.packages_distributions()

        # Multiple packages populate the `databricks` module namespace on Databricks; to avoid
        # bundling extraneous Databricks packages into model dependencies, we scope each module
        # to its relevant package
        _MODULES_TO_PACKAGES.update(DATABRICKS_MODULES_TO_PACKAGES)
        if "databricks" in _MODULES_TO_PACKAGES:
            _MODULES_TO_PACKAGES["databricks"] = [
                package
                for package in _MODULES_TO_PACKAGES["databricks"]
                if package not in _flatten(DATABRICKS_MODULES_TO_PACKAGES.values())
            ]

        # In Databricks, `_MODULES_TO_PACKAGES` doesn't contain pyspark since it's not installed
        # via pip or conda. To work around this issue, manually add pyspark.
        if is_in_databricks_runtime():
            _MODULES_TO_PACKAGES.update({"pyspark": ["pyspark"]})


def _init_packages_to_modules_map():
    _init_modules_to_packages_map()
    global _PACKAGES_TO_MODULES
    _PACKAGES_TO_MODULES = {}
    for module, pkg_list in _MODULES_TO_PACKAGES.items():
        for pkg_name in pkg_list:
            _PACKAGES_TO_MODULES[pkg_name] = module


# Represents the PyPI package index at a particular date
# :param date: The YYYY-MM-DD formatted string date on which the index was fetched.
# :param package_names: The set of package names in the index.
_PyPIPackageIndex = namedtuple("_PyPIPackageIndex", ["date", "package_names"])


def _load_pypi_package_index():
    with Path(mlflow.__file__).parent.joinpath("pypi_package_index.json").open() as f:
        index_dict = json.load(f)

    return _PyPIPackageIndex(
        date=index_dict["index_date"],
        package_names=set(index_dict["package_names"]),
    )


_PYPI_PACKAGE_INDEX = None


def _infer_requirements(model_uri, flavor):
    """
    Infers the pip requirements of the specified model by creating a subprocess and loading
    the model in it to determine which packages are imported.

    :param model_uri: The URI of the model.
    :param: flavor: The flavor name of the model.
    :return: A list of inferred pip requirements.
    """
    _init_modules_to_packages_map()
    global _PYPI_PACKAGE_INDEX
    if _PYPI_PACKAGE_INDEX is None:
        _PYPI_PACKAGE_INDEX = _load_pypi_package_index()

    modules = _capture_imported_modules(model_uri, flavor)
    packages = _flatten([_MODULES_TO_PACKAGES.get(module, []) for module in modules])
    packages = map(_normalize_package_name, packages)
    packages = _prune_packages(packages)
    excluded_packages = [
        # Certain packages (e.g. scikit-learn 0.24.2) imports `setuptools` or `pkg_resources`
        # (a module provided by `setuptools`) to process or interact with package metadata.
        # It should be safe to exclude `setuptools` because it's rare to encounter a python
        # environment where `setuptools` is not pre-installed.
        "setuptools",
        # Exclude a package that provides the mlflow module (e.g. mlflow, mlflow-skinny).
        # Certain flavors (e.g. pytorch) import mlflow while loading a model, but mlflow should
        # not be counted as a model requirement.
        *_MODULES_TO_PACKAGES.get("mlflow", []),
    ]
    packages = packages - set(excluded_packages)
    unrecognized_packages = packages - _PYPI_PACKAGE_INDEX.package_names
    if unrecognized_packages:
        _logger.warning(
            "The following packages were not found in the public PyPI package index as of"
            " %s; if these packages are not present in the public PyPI index, you must install"
            " them manually before loading your model: %s",
            _PYPI_PACKAGE_INDEX.date,
            unrecognized_packages,
        )
    return sorted(map(_get_pinned_requirement, packages))


def _get_local_version_label(version):
    """
    Extracts a local version label from `version`.

    :param version: A version string.
    """
    try:
        return Version(version).local
    except InvalidVersion:
        return None


def _strip_local_version_label(version):
    """
    Strips a local version label in `version`.

    Local version identifiers:
    https://www.python.org/dev/peps/pep-0440/#local-version-identifiers

    :param version: A version string to strip.
    """

    class IgnoreLocal(Version):
        @property
        def local(self):
            return None

    try:
        return str(IgnoreLocal(version))
    except InvalidVersion:
        return version


def _get_pinned_requirement(package, version=None, module=None):
    """
    Returns a string representing a pinned pip requirement to install the specified package and
    version (e.g. 'mlflow==1.2.3').

    :param package: The name of the package.
    :param version: The version of the package. If None, defaults to the installed version.
    :param module: The name of the top-level module provided by the package . For example,
                   if `package` is 'scikit-learn', `module` should be 'sklearn'. If None, defaults
                   to `package`.
    """
    if version is None:
        version_raw = _get_installed_version(package, module)
        local_version_label = _get_local_version_label(version_raw)
        if local_version_label:
            version = _strip_local_version_label(version_raw)
            if not (is_in_databricks_runtime() and package in ("torch", "torchvision")):
                msg = (
                    f"Found {package} version ({version_raw}) contains a local version label "
                    f"(+{local_version_label}). MLflow logged a pip requirement for this package "
                    f"as '{package}=={version}' without the local version label to make it "
                    "installable from PyPI. To specify pip requirements containing local version "
                    "labels, please use `conda_env` or `pip_requirements`."
                )
                _logger.warning(msg)

        else:
            version = version_raw

    return f"{package}=={version}"


class _MismatchedPackageInfo(NamedTuple):
    package_name: str
    installed_version: Optional[str]
    requirement: str

    def __str__(self):
        current_status = self.installed_version if self.installed_version else "uninstalled"
        return f"{self.package_name} (current: {current_status}, required: {self.requirement})"


def _check_requirement_satisfied(requirement_str):
    """
    Checks whether the current python environment satisfies the given requirement if it is parsable
    as a package name and a set of version specifiers, and returns a `_MismatchedPackageInfo`
    object containing the mismatched package name, installed version, and requirement if the
    requirement is not satisfied. Otherwise, returns None.
    """
    _init_packages_to_modules_map()
    try:
        req = Requirement(requirement_str)
    except Exception:
        # We reach here if the requirement string is a file path or a URL.
        # Extracting the package name from the requirement string is not trivial,
        # so we skip the check.
        return
    pkg_name = req.name

    try:
        installed_version = _get_installed_version(pkg_name, _PACKAGES_TO_MODULES.get(pkg_name))
    except ModuleNotFoundError:
        return _MismatchedPackageInfo(
            package_name=pkg_name,
            installed_version=None,
            requirement=requirement_str,
        )

    if (
        pkg_name == "mlflow"
        and installed_version == mlflow.__version__
        and Version(installed_version).is_devrelease
    ):
        return None

    if len(req.specifier) > 0 and not req.specifier.contains(installed_version):
        return _MismatchedPackageInfo(
            package_name=pkg_name,
            installed_version=installed_version,
            requirement=requirement_str,
        )

    return None
