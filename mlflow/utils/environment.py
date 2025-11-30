import hashlib
import importlib.metadata
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version

from mlflow.environment_variables import (
    _MLFLOW_ACTIVE_MODEL_ID,
    _MLFLOW_TESTING,
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT,
    MLFLOW_LOCK_MODEL_DEPENDENCIES,
    MLFLOW_REQUIREMENTS_INFERENCE_RAISE_ERRORS,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking import get_tracking_uri
from mlflow.tracking.fluent import _get_experiment_id, get_active_model_id
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.databricks_utils import (
    _get_databricks_serverless_env_vars,
    get_databricks_env_vars,
    is_databricks_connect,
    is_in_databricks_runtime,
)
from mlflow.utils.os import is_windows
from mlflow.utils.process import _exec_cmd
from mlflow.utils.requirements_utils import (
    _infer_requirements,
    _parse_requirements,
    warn_dependency_requirement_mismatches,
)
from mlflow.utils.timeout import MlflowTimeoutError, run_with_timeout
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)

_conda_header = """\
name: mlflow-env
channels:
  - conda-forge
"""

_CONDA_ENV_FILE_NAME = "conda.yaml"
_REQUIREMENTS_FILE_NAME = "requirements.txt"
_CONSTRAINTS_FILE_NAME = "constraints.txt"
_PYTHON_ENV_FILE_NAME = "python_env.yaml"


# Note this regular expression does not cover all possible patterns
_CONDA_DEPENDENCY_REGEX = re.compile(
    r"^(?P<package>python|pip|setuptools|wheel)"
    r"(?P<operator><|>|<=|>=|=|==|!=)?"
    r"(?P<version>[\d.]+)?$"
)


class _PythonEnv:
    BUILD_PACKAGES = ("pip", "setuptools", "wheel")

    def __init__(self, python=None, build_dependencies=None, dependencies=None):
        """
        Represents environment information for MLflow Models and Projects.

        Args:
            python: Python version for the environment. If unspecified, defaults to the current
                Python version.
            build_dependencies: List of build dependencies for the environment that must
                be installed before installing ``dependencies``. If unspecified,
                defaults to an empty list.
            dependencies: List of dependencies for the environment. If unspecified, defaults to
                an empty list.
        """
        if python is not None and not isinstance(python, str):
            raise TypeError(f"`python` must be a string but got {type(python)}")
        if build_dependencies is not None and not isinstance(build_dependencies, list):
            raise TypeError(
                f"`build_dependencies` must be a list but got {type(build_dependencies)}"
            )
        if dependencies is not None and not isinstance(dependencies, list):
            raise TypeError(f"`dependencies` must be a list but got {type(dependencies)}")

        self.python = python or PYTHON_VERSION
        self.build_dependencies = build_dependencies or []
        self.dependencies = dependencies or []

    def __str__(self):
        return str(self.to_dict())

    @classmethod
    def current(cls):
        return cls(
            python=PYTHON_VERSION,
            build_dependencies=cls.get_current_build_dependencies(),
            dependencies=[f"-r {_REQUIREMENTS_FILE_NAME}"],
        )

    @staticmethod
    def get_current_build_dependencies():
        build_dependencies = []
        for package in _PythonEnv.BUILD_PACKAGES:
            version = _get_package_version(package)
            dep = (package + "==" + version) if version else package
            build_dependencies.append(dep)
        return build_dependencies

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def to_yaml(self, path):
        with open(path, "w") as f:
            # Exclude None and empty lists
            data = {k: v for k, v in self.to_dict().items() if v}
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @staticmethod
    def get_dependencies_from_conda_yaml(path):
        with open(path) as f:
            conda_env = yaml.safe_load(f)

        python = None
        build_dependencies = None
        unmatched_dependencies = []
        dependencies = None
        for dep in conda_env.get("dependencies", []):
            if isinstance(dep, str):
                match = _CONDA_DEPENDENCY_REGEX.match(dep)
                if not match:
                    unmatched_dependencies.append(dep)
                    continue
                package = match.group("package")
                operator = match.group("operator")
                version = match.group("version")

                # Python
                if not python and package == "python":
                    if operator is None:
                        raise MlflowException.invalid_parameter_value(
                            f"Invalid dependency for python: {dep}. "
                            "It must be pinned (e.g. python=3.8.13)."
                        )

                    if operator in ("<", ">", "!="):
                        raise MlflowException(
                            f"Invalid version comparator for python: '{operator}'. "
                            "Must be one of ['<=', '>=', '=', '=='].",
                            error_code=INVALID_PARAMETER_VALUE,
                        )
                    python = version
                    continue

                # Build packages
                if build_dependencies is None:
                    build_dependencies = []
                # "=" is an invalid operator for pip
                operator = "==" if operator == "=" else operator
                build_dependencies.append(package + (operator or "") + (version or ""))
            elif _is_pip_deps(dep):
                dependencies = dep["pip"]
            else:
                raise MlflowException(
                    f"Invalid conda dependency: {dep}. Must be str or dict in the form of "
                    '{"pip": [...]}',
                    error_code=INVALID_PARAMETER_VALUE,
                )

        if python is None:
            _logger.warning(
                f"{path} does not include a python version specification. "
                f"Using the current python version {PYTHON_VERSION}."
            )
            python = PYTHON_VERSION

        if unmatched_dependencies:
            _logger.warning(
                "The following conda dependencies will not be installed in the resulting "
                "environment: %s",
                unmatched_dependencies,
            )

        return {
            "python": python,
            "build_dependencies": build_dependencies,
            "dependencies": dependencies,
        }

    @classmethod
    def from_conda_yaml(cls, path):
        return cls.from_dict(cls.get_dependencies_from_conda_yaml(path))


def _mlflow_conda_env(
    path=None,
    additional_conda_deps=None,
    additional_pip_deps=None,
    additional_conda_channels=None,
    install_mlflow=True,
):
    """Creates a Conda environment with the specified package channels and dependencies. If there
    are any pip dependencies, including from the install_mlflow parameter, then pip will be added to
    the conda dependencies. This is done to ensure that the pip inside the conda environment is
    used to install the pip dependencies.

    Args:
        path: Local filesystem path where the conda env file is to be written. If unspecified,
            the conda env will not be written to the filesystem; it will still be returned
            in dictionary format.
        additional_conda_deps: List of additional conda dependencies passed as strings.
        additional_pip_deps: List of additional pip dependencies passed as strings.
        additional_conda_channels: List of additional conda channels to search when resolving
            packages.

    Returns:
        None if path is specified. Otherwise, the a dictionary representation of the
        Conda environment.

    """
    additional_pip_deps = additional_pip_deps or []
    mlflow_deps = (
        [f"mlflow=={VERSION}"]
        if install_mlflow and not _contains_mlflow_requirement(additional_pip_deps)
        else []
    )
    pip_deps = mlflow_deps + additional_pip_deps
    conda_deps = additional_conda_deps or []
    if pip_deps:
        pip_version = _get_package_version("pip")
        if pip_version is not None:
            # When a new version of pip is released on PyPI, it takes a while until that version is
            # uploaded to conda-forge. This time lag causes `conda create` to fail with
            # a `ResolvePackageNotFound` error. As a workaround for this issue, use `<=` instead
            # of `==` so conda installs `pip_version - 1` when `pip_version` is unavailable.
            conda_deps.append(f"pip<={pip_version}")
        else:
            _logger.warning(
                "Failed to resolve installed pip version. ``pip`` will be added to conda.yaml"
                " environment spec without a version specifier."
            )
            conda_deps.append("pip")

    env = yaml.safe_load(_conda_header)
    env["dependencies"] = [f"python={PYTHON_VERSION}"]
    env["dependencies"] += conda_deps
    env["dependencies"].append({"pip": pip_deps})
    if additional_conda_channels is not None:
        env["channels"] += additional_conda_channels

    if path is not None:
        with open(path, "w") as out:
            yaml.safe_dump(env, stream=out, default_flow_style=False)
        return None
    else:
        return env


def _get_package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _mlflow_additional_pip_env(pip_deps, path=None):
    requirements = "\n".join(pip_deps)
    if path is not None:
        with open(path, "w") as out:
            out.write(requirements)
        return None
    else:
        return requirements


def _is_pip_deps(dep):
    """
    Returns True if `dep` is a dict representing pip dependencies
    """
    return isinstance(dep, dict) and "pip" in dep


def _get_pip_deps(conda_env):
    """
    Returns:
        The pip dependencies from the conda env.
    """
    if conda_env is not None:
        for dep in conda_env["dependencies"]:
            if _is_pip_deps(dep):
                return dep["pip"]
    return []


def _overwrite_pip_deps(conda_env, new_pip_deps):
    """
    Overwrites the pip dependencies section in the given conda env dictionary.

    {
        "name": "env",
        "channels": [...],
        "dependencies": [
            ...,
            "pip",
            {"pip": [...]},  <- Overwrite this
        ],
    }
    """
    deps = conda_env.get("dependencies", [])
    new_deps = []
    contains_pip_deps = False
    for dep in deps:
        if _is_pip_deps(dep):
            contains_pip_deps = True
            new_deps.append({"pip": new_pip_deps})
        else:
            new_deps.append(dep)

    if not contains_pip_deps:
        new_deps.append({"pip": new_pip_deps})

    return {**conda_env, "dependencies": new_deps}


def _log_pip_requirements(conda_env, path, requirements_file=_REQUIREMENTS_FILE_NAME):
    pip_deps = _get_pip_deps(conda_env)
    _mlflow_additional_pip_env(pip_deps, path=os.path.join(path, requirements_file))


def _parse_pip_requirements(pip_requirements):
    """Parses an iterable of pip requirement strings or a pip requirements file.

    Args:
        pip_requirements: Either an iterable of pip requirement strings
            (e.g. ``["scikit-learn", "-r requirements.txt"]``) or the string path to a pip
            requirements file on the local filesystem (e.g. ``"requirements.txt"``). If ``None``,
            an empty list will be returned.

    Returns:
        A tuple of parsed requirements and constraints.
    """
    if pip_requirements is None:
        return [], []

    def _is_string(x):
        return isinstance(x, str)

    def _is_iterable(x):
        try:
            iter(x)
            return True
        except Exception:
            return False

    if _is_string(pip_requirements):
        with open(pip_requirements) as f:
            return _parse_pip_requirements(f.read().splitlines())
    elif _is_iterable(pip_requirements) and all(map(_is_string, pip_requirements)):
        requirements = []
        constraints = []
        for req_or_con in _parse_requirements(pip_requirements, is_constraint=False):
            if req_or_con.is_constraint:
                constraints.append(req_or_con.req_str)
            else:
                requirements.append(req_or_con.req_str)

        return requirements, constraints
    else:
        raise TypeError(
            "`pip_requirements` must be either a string path to a pip requirements file on the "
            "local filesystem or an iterable of pip requirement strings, but got `{}`".format(
                type(pip_requirements)
            )
        )


_INFER_PIP_REQUIREMENTS_GENERAL_ERROR_MESSAGE = (
    "Encountered an unexpected error while inferring pip requirements "
    "(model URI: {model_uri}, flavor: {flavor}). Fall back to return {fallback}. "
    "Set logging level to DEBUG to see the full traceback. "
)


def infer_pip_requirements(model_uri, flavor, fallback=None, timeout=None, extra_env_vars=None):
    """Infers the pip requirements of the specified model by creating a subprocess and loading
    the model in it to determine which packages are imported.

    Args:
        model_uri: The URI of the model.
        flavor: The flavor name of the model.
        fallback: If provided, an unexpected error during the inference procedure is swallowed
            and the value of ``fallback`` is returned. Otherwise, the error is raised.
        timeout: If specified, the inference operation is bound by the timeout (in seconds).
        extra_env_vars: A dictionary of extra environment variables to pass to the subprocess.
            Default to None.

    Returns:
        A list of inferred pip requirements (e.g. ``["scikit-learn==0.24.2", ...]``).

    """
    raise_on_error = MLFLOW_REQUIREMENTS_INFERENCE_RAISE_ERRORS.get()

    if timeout and is_windows():
        timeout = None
        _logger.warning(
            "On Windows, timeout is not supported for model requirement inference. Therefore, "
            "the operation is not bound by a timeout and may hang indefinitely. If it hangs, "
            "please consider specifying the signature manually."
        )

    try:
        if timeout:
            with run_with_timeout(timeout):
                return _infer_requirements(
                    model_uri, flavor, raise_on_error=raise_on_error, extra_env_vars=extra_env_vars
                )
        else:
            return _infer_requirements(
                model_uri, flavor, raise_on_error=raise_on_error, extra_env_vars=extra_env_vars
            )
    except Exception as e:
        if raise_on_error or (fallback is None):
            raise

        if isinstance(e, MlflowTimeoutError):
            msg = (
                "Attempted to infer pip requirements for the saved model or pipeline but the "
                f"operation timed out in {timeout} seconds. Fall back to return {fallback}. "
                "You can specify a different timeout by setting the environment variable "
                f"{MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT}."
            )
        else:
            msg = _INFER_PIP_REQUIREMENTS_GENERAL_ERROR_MESSAGE.format(
                model_uri=model_uri, flavor=flavor, fallback=fallback
            )
        _logger.warning(msg)
        _logger.debug("", exc_info=True)
        return fallback


def _get_uv_options_for_databricks() -> tuple[list[str], dict[str, str]] | None:
    """
    Retrieves the predefined secrets to configure `pip` for Databricks, and converts them into
    command-line arguments and environment variables for `uv`.

    References:
    - https://docs.databricks.com/aws/en/compute/serverless/dependencies#predefined-secret-scope-name
    - https://docs.astral.sh/uv/configuration/environment/#environment-variables
    """
    from databricks.sdk import WorkspaceClient

    from mlflow.utils.databricks_utils import (
        _get_dbutils,
        _NoDbutilsError,
        is_in_databricks_runtime,
    )

    if not is_in_databricks_runtime():
        return None

    workspace_client = WorkspaceClient()
    secret_scopes = workspace_client.secrets.list_scopes()
    if not any(s.name == "databricks-package-management" for s in secret_scopes):
        return None

    try:
        dbutils = _get_dbutils()
    except _NoDbutilsError:
        return None

    def get_secret(key: str) -> str | None:
        """
        Retrieves a secret from the Databricks secrets scope.
        """
        try:
            return dbutils.secrets.get(scope="databricks-package-management", key=key)
        except Exception as e:
            _logger.debug(f"Failed to fetch secret '{key}': {e}")
            return None

    args: list[str] = []
    if url := get_secret("pip-index-url"):
        args.append(f"--index-url={url}")

    if urls := get_secret("pip-extra-index-urls"):
        args.append(f"--extra-index-url={urls}")

    # There is no command-line option for SSL_CERT_FILE in `uv`.
    envs: dict[str, str] = {}
    if cert := get_secret("pip-cert"):
        envs["SSL_CERT_FILE"] = cert

    _logger.debug(f"uv arguments and environment variables: {args}, {envs}")
    return args, envs


def _lock_requirements(
    requirements: list[str], constraints: list[str] | None = None
) -> list[str] | None:
    """
    Locks the given requirements using `uv`. Returns the locked requirements when the locking is
    performed successfully, otherwise returns None.
    """
    if not MLFLOW_LOCK_MODEL_DEPENDENCIES.get():
        return None

    uv_bin = shutil.which("uv")
    if uv_bin is None:
        _logger.debug("`uv` binary not found. Skipping locking requirements.")
        return None

    _logger.info("Locking requirements...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = pathlib.Path(tmp_dir)
        in_file = tmp_dir_path / "requirements.in"
        in_file.write_text("\n".join(requirements))
        out_file = tmp_dir_path / "requirements.out"
        constraints_opt: list[str] = []
        if constraints:
            constraints_file = tmp_dir_path / "constraints.txt"
            constraints_file.write_text("\n".join(constraints))
            constraints_opt = [f"--constraints={constraints_file}"]
        elif pip_constraint := os.environ.get("PIP_CONSTRAINT"):
            # If PIP_CONSTRAINT is set, use it as a constraint file
            constraints_opt = [f"--constraints={pip_constraint}"]

        try:
            if res := _get_uv_options_for_databricks():
                uv_options, uv_envs = res
            else:
                uv_options = []
                uv_envs = {}
            out = subprocess.check_output(
                [
                    uv_bin,
                    "pip",
                    "compile",
                    "--color=never",
                    "--universal",
                    "--no-annotate",
                    "--no-header",
                    f"--python-version={PYTHON_VERSION}",
                    f"--output-file={out_file}",
                    *uv_options,
                    *constraints_opt,
                    in_file,
                ],
                stderr=subprocess.STDOUT,
                env=os.environ.copy() | uv_envs,
                text=True,
            )
            _logger.debug(f"Successfully compiled requirements with `uv`:\n{out}")
        except subprocess.CalledProcessError as e:
            _logger.warning(f"Failed to lock requirements:\n{e.output}")
            return None

        return [
            "# Original requirements",
            *(f"# {l}" for l in requirements),  # Preserve original requirements as comments
            "#",
            "# Locked requirements",
            *out_file.read_text().splitlines(),
        ]


def _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements):
    """
    Validates that only one or none of `conda_env`, `pip_requirements`, and
    `extra_pip_requirements` is specified.
    """
    args = [
        conda_env,
        pip_requirements,
        extra_pip_requirements,
    ]
    specified = [arg for arg in args if arg is not None]
    if len(specified) > 1:
        raise ValueError(
            "Only one of `conda_env`, `pip_requirements`, and "
            "`extra_pip_requirements` can be specified"
        )


# PIP requirement parser inspired from https://github.com/pypa/pip/blob/b392833a0f1cff1bbee1ac6dbe0270cccdd0c11f/src/pip/_internal/req/req_file.py#L400
def _get_pip_requirement_specifier(requirement_string):
    tokens = requirement_string.split(" ")
    for idx, token in enumerate(tokens):
        if token.startswith("-"):
            return " ".join(tokens[:idx])
    return requirement_string


def _is_mlflow_requirement(requirement_string):
    """
    Returns True if `requirement_string` represents a requirement for mlflow (e.g. 'mlflow==1.2.3').
    """
    # "/opt/mlflow" is the path where we mount the mlflow source code in the Docker container
    # when running tests.
    if _MLFLOW_TESTING.get() and requirement_string == "/opt/mlflow":
        return True

    try:
        # `Requirement` throws an `InvalidRequirement` exception if `requirement_string` doesn't
        # conform to PEP 508 (https://www.python.org/dev/peps/pep-0508).
        return Requirement(requirement_string).name.lower() in [
            "mlflow",
            "mlflow-skinny",
            "mlflow-tracing",
        ]
    except InvalidRequirement:
        # A local file path or URL falls into this branch.

        # `Requirement` throws an `InvalidRequirement` exception if `requirement_string` contains
        # per-requirement options (ex: package hashes)
        # GitHub issue: https://github.com/pypa/packaging/issues/488
        # Per-requirement-option spec: https://pip.pypa.io/en/stable/reference/requirements-file-format/#per-requirement-options
        requirement_specifier = _get_pip_requirement_specifier(requirement_string)
        try:
            # Try again with the per-requirement options removed
            return Requirement(requirement_specifier).name.lower() == "mlflow"
        except InvalidRequirement:
            # Support defining branch dependencies for local builds or direct GitHub builds
            # from source.
            # Example: mlflow @ git+https://github.com/mlflow/mlflow@branch_2.0
            repository_matches = ["/mlflow", "mlflow@git"]

            return any(
                match in requirement_string.replace(" ", "").lower() for match in repository_matches
            )


def _generate_mlflow_version_pinning() -> str:
    """Returns a pinned requirement for the current MLflow version (e.g., "mlflow==3.2.1").

    Returns:
        A pinned requirement for the current MLflow version.

    """
    if _MLFLOW_TESTING.get():
        # The local PyPI server should be running. It serves a wheel for the current MLflow version.
        return f"mlflow=={VERSION}"

    version = Version(VERSION)
    if not version.is_devrelease:
        # mlflow is installed from PyPI.
        return f"mlflow=={VERSION}"

    # We reach here when mlflow is installed from the source outside of the MLflow CI environment
    # (e.g., Databricks notebook).

    # mlflow installed from the source for development purposes. A dev version (e.g., 2.8.1.dev0)
    # is always a micro-version ahead of the latest release (unless it's manually modified)
    # and can't be installed from PyPI. We therefore subtract 1 from the micro version when running
    # tests.
    return f"mlflow=={version.major}.{version.minor}.{version.micro - 1}"


def _contains_mlflow_requirement(requirements):
    """
    Returns True if `requirements` contains a requirement for mlflow (e.g. 'mlflow==1.2.3').
    """
    return any(map(_is_mlflow_requirement, requirements))


def _process_pip_requirements(
    default_pip_requirements, pip_requirements=None, extra_pip_requirements=None
):
    """
    Processes `pip_requirements` and `extra_pip_requirements` passed to `mlflow.*.save_model` or
    `mlflow.*.log_model`, and returns a tuple of (conda_env, pip_requirements, pip_constraints).
    """
    constraints = []
    if pip_requirements is not None:
        pip_reqs, constraints = _parse_pip_requirements(pip_requirements)
    elif extra_pip_requirements is not None:
        extra_pip_requirements, constraints = _parse_pip_requirements(extra_pip_requirements)
        pip_reqs = default_pip_requirements + extra_pip_requirements
    else:
        pip_reqs = default_pip_requirements

    if not _contains_mlflow_requirement(pip_reqs):
        pip_reqs.insert(0, _generate_mlflow_version_pinning())

    sanitized_pip_reqs = _deduplicate_requirements(pip_reqs)
    sanitized_pip_reqs = _remove_incompatible_requirements(sanitized_pip_reqs)

    # Check if pip requirements contain incompatible version with the current environment
    warn_dependency_requirement_mismatches(sanitized_pip_reqs)

    if locked_requirements := _lock_requirements(sanitized_pip_reqs, constraints):
        # Locking requirements was performed successfully
        sanitized_pip_reqs = locked_requirements
    else:
        # Locking requirements was skipped or failed
        if constraints:
            sanitized_pip_reqs.append(f"-c {_CONSTRAINTS_FILE_NAME}")

    # Set `install_mlflow` to False because `pip_reqs` already contains `mlflow`
    conda_env = _mlflow_conda_env(additional_pip_deps=sanitized_pip_reqs, install_mlflow=False)
    return conda_env, sanitized_pip_reqs, constraints


def _deduplicate_requirements(requirements):
    """
    De-duplicates a list of pip package requirements, handling complex scenarios such as merging
    extras and combining version constraints.

    This function processes a list of pip package requirements and de-duplicates them. It handles
    standard PyPI packages and non-standard requirements (like URLs or local paths). The function
    merges extras and combines version constraints for duplicate packages. The most restrictive
    version specifications or the ones with extras are prioritized. If incompatible version
    constraints are detected, it raises an MlflowException.

    Args:
        requirements (list of str): A list of pip package requirement strings.

    Returns:
        list of str: A deduplicated list of pip package requirements.

    Raises:
        MlflowException: If incompatible version constraints are detected among the provided
                         requirements.

    Examples:
        - Input: ["packageA", "packageA==1.0"]
          Output: ["packageA==1.0"]

        - Input: ["packageX>1.0", "packageX[extras]", "packageX<2.0"]
          Output: ["packageX[extras]>1.0,<2.0"]

        - Input: ["markdown[extra1]>=3.5.1", "markdown[extra2]<4", "markdown"]
          Output: ["markdown[extra1,extra2]>=3.5.1,<4"]

        - Input: ["scikit-learn==1.1", "scikit-learn<1"]
          Raises MlflowException indicating incompatible versions.

    Note:
        - Non-standard requirements (like URLs or file paths) are included as-is.
        - If a requirement appears multiple times with different sets of extras, they are merged.
        - The function uses `_validate_version_constraints` to check for incompatible version
          constraints by doing a dry-run pip install of a requirements collection.
    """
    deduped_reqs = {}

    for req in requirements:
        try:
            parsed_req = Requirement(req)
            base_pkg = parsed_req.name

            existing_req = deduped_reqs.get(base_pkg)

            if not existing_req:
                deduped_reqs[base_pkg] = parsed_req
            else:
                # Verify that there are not unresolvable constraints applied if set and combine
                # if possible
                if (
                    existing_req.specifier
                    and parsed_req.specifier
                    and existing_req.specifier != parsed_req.specifier
                ):
                    _validate_version_constraints([str(existing_req), req])
                    parsed_req.specifier = ",".join(
                        [str(existing_req.specifier), str(parsed_req.specifier)]
                    )

                # Preserve existing specifiers
                if existing_req.specifier and not parsed_req.specifier:
                    parsed_req.specifier = existing_req.specifier

                # Combine and apply extras if specified
                if (
                    existing_req.extras
                    and parsed_req.extras
                    and existing_req.extras != parsed_req.extras
                ):
                    parsed_req.extras = sorted(set(existing_req.extras).union(parsed_req.extras))
                elif existing_req.extras and not parsed_req.extras:
                    parsed_req.extras = existing_req.extras

                deduped_reqs[base_pkg] = parsed_req

        except InvalidRequirement:
            # Include non-standard package strings as-is
            if req not in deduped_reqs:
                deduped_reqs[req] = req
    return [str(req) for req in deduped_reqs.values()]


def _parse_requirement_name(req: str) -> str:
    try:
        return Requirement(req).name
    except InvalidRequirement:
        return req


def _remove_incompatible_requirements(requirements: list[str]) -> list[str]:
    req_names = {_parse_requirement_name(req) for req in requirements}
    if "databricks-connect" in req_names and req_names.intersection({"pyspark", "pyspark-connect"}):
        _logger.debug(
            "Found incompatible requirements: 'databricks-connect' with 'pyspark' or "
            "'pyspark-connect'. Removing 'pyspark' or 'pyspark-connect' from the requirements."
        )
        requirements = [
            req
            for req in requirements
            if _parse_requirement_name(req) not in ["pyspark", "pyspark-connect"]
        ]
    return requirements


def _validate_version_constraints(requirements):
    """
    Validates the version constraints of given Python package requirements using pip's resolver with
    the `--dry-run` option enabled that performs validation only (will not install packages).

    This function writes the requirements to a temporary file and then attempts to resolve
    them using pip's `--dry-run` install option. If any version conflicts are detected, it
    raises an MlflowException with details of the conflict.

    Args:
        requirements (list of str): A list of package requirements (e.g., `["pandas>=1.15",
        "pandas<2"]`).

    Raises:
        MlflowException: If any version conflicts are detected among the provided requirements.

    Returns:
        None: This function does not return anything. It either completes successfully or raises
        an MlflowException.

    Example:
        _validate_version_constraints(["tensorflow<2.0", "tensorflow>2.3"])
        # This will raise an exception due to boundary validity.
    """
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp_file:
        tmp_file.write("\n".join(requirements))
        tmp_file_name = tmp_file.name

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "-r", tmp_file_name],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise MlflowException.invalid_parameter_value(
            "The specified requirements versions are incompatible. Detected "
            f"conflicts: \n{e.stderr.decode()}"
        )
    finally:
        os.remove(tmp_file_name)


def _process_conda_env(conda_env):
    """
    Processes `conda_env` passed to `mlflow.*.save_model` or `mlflow.*.log_model`, and returns
    a tuple of (conda_env, pip_requirements, pip_constraints).
    """
    if isinstance(conda_env, str):
        with open(conda_env) as f:
            conda_env = yaml.safe_load(f)
    elif not isinstance(conda_env, dict):
        raise TypeError(
            "Expected a string path to a conda env yaml file or a `dict` representing a conda env, "
            f"but got `{type(conda_env).__name__}`"
        )

    # User-specified `conda_env` may contain requirements/constraints file references
    pip_reqs = _get_pip_deps(conda_env)
    pip_reqs, constraints = _parse_pip_requirements(pip_reqs)
    if not _contains_mlflow_requirement(pip_reqs):
        pip_reqs.insert(0, _generate_mlflow_version_pinning())

    # Check if pip requirements contain incompatible version with the current environment
    warn_dependency_requirement_mismatches(pip_reqs)

    if constraints:
        pip_reqs.append(f"-c {_CONSTRAINTS_FILE_NAME}")

    conda_env = _overwrite_pip_deps(conda_env, pip_reqs)
    return conda_env, pip_reqs, constraints


def _get_mlflow_env_name(s):
    """Creates an environment name for an MLflow model by hashing the given string.

    Args:
        s: String to hash (e.g. the content of `conda.yaml`).

    Returns:
        String in the form of "mlflow-{hash}"
        (e.g. "mlflow-da39a3ee5e6b4b0d3255bfef95601890afd80709")

    """
    return "mlflow-" + hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()


def _get_pip_install_mlflow():
    """
    Returns a command to pip-install mlflow. If the MLFLOW_HOME environment variable exists,
    returns "pip install -e {MLFLOW_HOME} 1>&2", otherwise
    "pip install mlflow=={mlflow.__version__} 1>&2".
    """
    if mlflow_home := os.getenv("MLFLOW_HOME"):  # dev version
        return f"pip install -e {mlflow_home} 1>&2"
    else:
        return f"pip install mlflow=={VERSION} 1>&2"


def _get_requirements_from_file(
    file_path: pathlib.Path,
) -> list[Requirement]:
    data = file_path.read_text()
    if file_path.name == _CONDA_ENV_FILE_NAME:
        conda_env = yaml.safe_load(data)
        reqs = _get_pip_deps(conda_env)
    else:
        reqs = data.splitlines()
    return [Requirement(req) for req in reqs if req]


def _write_requirements_to_file(
    file_path: pathlib.Path,
    new_reqs: list[str],
) -> None:
    if file_path.name == _CONDA_ENV_FILE_NAME:
        conda_env = yaml.safe_load(file_path.read_text())
        conda_env = _overwrite_pip_deps(conda_env, new_reqs)
        with file_path.open("w") as file:
            yaml.dump(conda_env, file)
    else:
        file_path.write_text("\n".join(new_reqs))


def _add_or_overwrite_requirements(
    new_reqs: list[Requirement],
    old_reqs: list[Requirement],
) -> list[str]:
    deduped_new_reqs = _deduplicate_requirements([str(req) for req in new_reqs])
    deduped_new_reqs = [Requirement(req) for req in deduped_new_reqs]

    old_reqs_dict = {req.name: str(req) for req in old_reqs}
    new_reqs_dict = {req.name: str(req) for req in deduped_new_reqs}
    old_reqs_dict.update(new_reqs_dict)
    return list(old_reqs_dict.values())


def _remove_requirements(
    reqs_to_remove: list[Requirement],
    old_reqs: list[Requirement],
) -> list[str]:
    old_reqs_dict = {req.name: str(req) for req in old_reqs}
    for req in reqs_to_remove:
        if req.name not in old_reqs_dict:
            _logger.warning(f'"{req.name}" not found in requirements, ignoring')
        old_reqs_dict.pop(req.name, None)
    return list(old_reqs_dict.values())


class Environment:
    def __init__(self, activate_cmd, extra_env=None):
        if not isinstance(activate_cmd, list):
            activate_cmd = [activate_cmd]
        self._activate_cmd = activate_cmd
        self._extra_env = extra_env or {}

    def get_activate_command(self):
        return self._activate_cmd

    def execute(
        self,
        command,
        command_env=None,
        preexec_fn=None,
        capture_output=False,
        stdout=None,
        stderr=None,
        stdin=None,
        synchronous=True,
    ):
        command_env = os.environ.copy() if command_env is None else deepcopy(command_env)
        if is_in_databricks_runtime():
            command_env.update(get_databricks_env_vars(get_tracking_uri()))
        if is_databricks_connect():
            command_env.update(_get_databricks_serverless_env_vars())
        if exp_id := _get_experiment_id():
            command_env[MLFLOW_EXPERIMENT_ID.name] = exp_id
        if active_model_id := get_active_model_id():
            command_env[_MLFLOW_ACTIVE_MODEL_ID.name] = active_model_id
        command_env.update(self._extra_env)
        if not isinstance(command, list):
            command = [command]

        separator = " && " if not is_windows() else " & "

        command = separator.join(map(str, self._activate_cmd + command))
        command = ["bash", "-c", command] if not is_windows() else ["cmd", "/c", command]
        _logger.info("=== Running command '%s'", command)
        return _exec_cmd(
            command,
            env=command_env,
            capture_output=capture_output,
            synchronous=synchronous,
            preexec_fn=preexec_fn,
            close_fds=True,
            stdout=stdout,
            stderr=stderr,
            stdin=stdin,
        )
