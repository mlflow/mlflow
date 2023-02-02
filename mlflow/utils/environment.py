import yaml
import os
import logging
import re
import hashlib
from packaging.requirements import Requirement, InvalidRequirement
from packaging.version import Version

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.process import _exec_cmd
from mlflow.utils.requirements_utils import (
    _parse_requirements,
    _infer_requirements,
)
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

_IS_UNIX = os.name != "nt"


class _PythonEnv:
    BUILD_PACKAGES = ("pip", "setuptools", "wheel")

    def __init__(self, python=None, build_dependencies=None, dependencies=None):
        """
        Represents environment information for MLflow Models and Projects.

        :param python: Python version for the environment. If unspecified, defaults to the current
                       Python version.
        :param build_dependencies: List of build dependencies for the environment that must
                                   be installed before installing ``dependencies``. If unspecified,
                                   defaults to an empty list.
        :param dependencies: List of dependencies for the environment. If unspecified, defaults to
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
    def _get_package_version(package_name):
        try:
            return __import__(package_name).__version__
        except (ImportError, AttributeError):
            return None

    @staticmethod
    def get_current_build_dependencies():
        build_dependencies = []
        for package in _PythonEnv.BUILD_PACKAGES:
            version = _PythonEnv._get_package_version(package)
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
    """
    Creates a Conda environment with the specified package channels and dependencies. If there are
    any pip dependencies, including from the install_mlflow parameter, then pip will be added to
    the conda dependencies. This is done to ensure that the pip inside the conda environment is
    used to install the pip dependencies.

    :param path: Local filesystem path where the conda env file is to be written. If unspecified,
                 the conda env will not be written to the filesystem; it will still be returned
                 in dictionary format.
    :param additional_conda_deps: List of additional conda dependencies passed as strings.
    :param additional_pip_deps: List of additional pip dependencies passed as strings.
    :param additional_conda_channels: List of additional conda channels to search when resolving
                                      packages.
    :return: ``None`` if ``path`` is specified. Otherwise, the a dictionary representation of the
             Conda environment.
    """
    pip_deps = (["mlflow"] if install_mlflow else []) + (
        additional_pip_deps if additional_pip_deps else []
    )
    conda_deps = additional_conda_deps if additional_conda_deps else []
    if pip_deps:
        pip_version = _get_pip_version()
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


def _get_pip_version():
    """
    :return: The version of ``pip`` that is installed in the current environment,
             or ``None`` if ``pip`` is not currently installed / does not have a
             ``__version__`` attribute.
    """
    try:
        import pip

        return pip.__version__
    except ImportError:
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
    :return: The pip dependencies from the conda env
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
    """
    Parses an iterable of pip requirement strings or a pip requirements file.

    :param pip_requirements: Either an iterable of pip requirement strings
        (e.g. ``["scikit-learn", "-r requirements.txt"]``) or the string path to a pip requirements
        file on the local filesystem (e.g. ``"requirements.txt"``). If ``None``, an empty list will
        be returned.
    :return: A tuple of parsed requirements and constraints.
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


_INFER_PIP_REQUIREMENTS_FALLBACK_MESSAGE = (
    "Encountered an unexpected error while inferring pip requirements (model URI: %s, flavor: %s),"
    " fall back to return %s. Set logging level to DEBUG to see the full traceback."
)


def infer_pip_requirements(model_uri, flavor, fallback=None):
    """
    Infers the pip requirements of the specified model by creating a subprocess and loading
    the model in it to determine which packages are imported.

    :param model_uri: The URI of the model.
    :param flavor: The flavor name of the model.
    :param fallback: If provided, an unexpected error during the inference procedure is swallowed
                     and the value of ``fallback`` is returned. Otherwise, the error is raised.
    :return: A list of inferred pip requirements (e.g. ``["scikit-learn==0.24.2", ...]``).
    """
    try:
        return _infer_requirements(model_uri, flavor)
    except Exception:
        if fallback is not None:
            _logger.warning(_INFER_PIP_REQUIREMENTS_FALLBACK_MESSAGE, model_uri, flavor, fallback)
            _logger.debug("", exc_info=True)
            return fallback
        raise


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
    try:
        # `Requirement` throws an `InvalidRequirement` exception if `requirement_string` doesn't
        # conform to PEP 508 (https://www.python.org/dev/peps/pep-0508).
        return Requirement(requirement_string).name.lower() in ["mlflow", "mlflow-skinny"]
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


def _generate_mlflow_version_pinning():
    """
    Determines the current MLflow version that is installed and adds a pinned boundary version range
    for mlflow. The upper bound is a cap on the next major revision. The lower bound is a cap on
    the current installed minor version(i.e., 'mlflow<3,>=2.1')
    :return: string for MLflow dependency version
    """
    mlflow_version = Version(VERSION)
    current_major_version = mlflow_version.major
    current_minor_version = mlflow_version.minor
    range_version = (
        f"mlflow<{current_major_version + 1},>={current_major_version}.{current_minor_version}"
    )
    return range_version


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

    if constraints:
        pip_reqs.append(f"-c {_CONSTRAINTS_FILE_NAME}")

    # Set `install_mlflow` to False because `pip_reqs` already contains `mlflow`
    conda_env = _mlflow_conda_env(additional_pip_deps=pip_reqs, install_mlflow=False)
    return conda_env, pip_reqs, constraints


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
            "but got `{}`".format(type(conda_env).__name__)
        )

    # User-specified `conda_env` may contain requirements/constraints file references
    pip_reqs = _get_pip_deps(conda_env)
    pip_reqs, constraints = _parse_pip_requirements(pip_reqs)

    if not _contains_mlflow_requirement(pip_reqs):
        pip_reqs.insert(0, _generate_mlflow_version_pinning())

    if constraints:
        pip_reqs.append(f"-c {_CONSTRAINTS_FILE_NAME}")

    conda_env = _overwrite_pip_deps(conda_env, pip_reqs)
    return conda_env, pip_reqs, constraints


def _get_mlflow_env_name(s):
    """
    Creates an environment name for an MLflow model by hashing the given string.

    :param s: String to hash (e.g. the content of `conda.yaml`).
    :returns: String in the form of "mlflow-{hash}"
              (e.g. "mlflow-da39a3ee5e6b4b0d3255bfef95601890afd80709")
    """
    return "mlflow-" + hashlib.sha1(s.encode("utf-8")).hexdigest()


def _get_pip_install_mlflow():
    """
    Returns a command to pip-install mlflow. If the MLFLOW_HOME environment variable exists,
    returns "pip install -e {MLFLOW_HOME} 1>&2", otherwise
    "pip install mlflow=={mlflow.__version__} 1>&2".
    """
    mlflow_home = os.getenv("MLFLOW_HOME")
    if mlflow_home:  # dev version
        return f"pip install -e {mlflow_home} 1>&2"
    else:
        return f"pip install mlflow=={VERSION} 1>&2"


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
        if command_env is None:
            command_env = os.environ.copy()
        command_env = {**self._extra_env, **command_env}
        if not isinstance(command, list):
            command = [command]

        if _IS_UNIX:
            separator = " && "
        else:
            separator = " & "

        command = separator.join(map(str, self._activate_cmd + command))
        if _IS_UNIX:
            command = ["bash", "-c", command]
        else:
            command = ["cmd", "/c", command]
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
