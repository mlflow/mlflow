import inspect
import os
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar

from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY


def _is_ci_env() -> bool:
    """
    Check if the current environment is a CI environment.
    If so, we should not track telemetry.
    """
    env_vars = {
        "PYTEST_CURRENT_TEST",  # https://docs.pytest.org/en/stable/example/simple.html#pytest-current-test-environment-variable
        "GITHUB_ACTIONS",  # https://docs.github.com/en/actions/reference/variables-reference?utm_source=chatgpt.com#default-environment-variables
        "CI",  # set by many CI providers
        "CIRCLECI",  # https://circleci.com/docs/variables/#built-in-environment-variables
        "GITLAB_CI",  # https://docs.gitlab.com/ci/variables/predefined_variables/#predefined-variables
        "JENKINS_URL",  # https://www.jenkins.io/doc/book/pipeline/jenkinsfile/#using-environment-variables
        "TRAVIS",  # https://docs.travis-ci.com/user/environment-variables/#default-environment-variables
        "TF_BUILD",  # https://learn.microsoft.com/en-us/azure/devops/pipelines/build/variables?view=azure-devops&tabs=yaml#system-variables
        "BITBUCKET_BUILD_NUMBER",  # https://support.atlassian.com/bitbucket-cloud/docs/variables-and-secrets/
        "CODEBUILD_BUILD_ARN",  # https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-env-vars.html
        "BUILDKITE",  # https://buildkite.com/docs/pipelines/configure/environment-variables
        # TODO: add runbot env var
    }
    if any(os.environ.get(var, "false").lower() == "true" for var in env_vars):
        return True
    return False


# NB: implement the function here to avoid unnecessary imports inside databricks_utils
def _is_in_databricks() -> bool:
    _DATABRICKS_VERSION_FILE_PATH = "/databricks/DBR_VERSION"

    # check if in databricks runtime
    version = os.environ.get("DATABRICKS_RUNTIME_VERSION")
    if version is None and os.path.exists(_DATABRICKS_VERSION_FILE_PATH):
        # In Databricks DCS cluster, it doesn't have DATABRICKS_RUNTIME_VERSION
        # environment variable, we have to read version from the version file.
        with open(_DATABRICKS_VERSION_FILE_PATH) as f:
            version = f.read().strip()
    if version is not None:
        return True

    # enable for databricks serving environment since it's a standalone environment
    # and we can track tracing events

    return False


def is_telemetry_disabled() -> bool:
    return (
        MLFLOW_DISABLE_TELEMETRY.get()
        or os.environ.get("DO_NOT_TRACK", "false").lower() == "true"
        or _is_ci_env()
        or _is_in_databricks()
    )


def _get_whitelist() -> dict[str, set[str]]:
    """
    Whitelist for APIs that are only invoked by MLflow but should be tracked.
    """
    whitelist = defaultdict(set)
    try:
        from mlflow.pyfunc.utils.data_validation import _infer_schema_from_list_type_hint

        whitelist[_infer_schema_from_list_type_hint.__module__].add(
            _infer_schema_from_list_type_hint.__qualname__
        )
    except ImportError:
        pass

    return whitelist


def should_skip_telemetry(func) -> bool:
    # If the function is in whitelist, we should always track it
    if func.__qualname__ in _get_whitelist().get(func.__module__, set()):
        return False

    if _disable_telemetry_tracking_var.get():
        return True

    frame = inspect.currentframe()
    try:
        # skip the current frame and the API call frames
        frame = frame.f_back.f_back if frame and frame.f_back else None
        module = inspect.getmodule(frame)
        # TODO: consider recording if this comes from databricks modules
        return module and module.__name__.startswith("mlflow")
    finally:
        del frame


# ContextVar to disable telemetry tracking in the current thread.
# This is thread-local to avoid race conditions when multiple threads are running in parallel.
# NB: this doesn't work if a nested function spawns a new thread (e.g. mlflow.genai.evaluate)
_disable_telemetry_tracking_var = ContextVar("disable_telemetry_tracking", default=False)


@contextmanager
def _disable_telemetry():
    """
    Context manager to disable telemetry tracking in the following scenarios:
    1. Circular API calls: When MLflow invokes `databricks-agents` APIs, which in turn call back
        into MLflow APIs. This prevents telemetry from tracking internal, nested invocations.
    2. Code-based model logging: During model logging, the model file may be executed directly,
        potentially triggering additional telemetry logging inside model file. This context
        suppresses such telemetry during model loading and logging.
    """
    token = _disable_telemetry_tracking_var.set(True)
    try:
        yield
    finally:
        _disable_telemetry_tracking_var.reset(token)
