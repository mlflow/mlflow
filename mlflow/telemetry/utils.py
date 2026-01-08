import logging
import os
from typing import Any

import requests
from packaging.version import Version

from mlflow.environment_variables import (
    _MLFLOW_TELEMETRY_LOGGING,
    _MLFLOW_TESTING_TELEMETRY,
    MLFLOW_DISABLE_TELEMETRY,
)
from mlflow.telemetry.constant import (
    CONFIG_STAGING_URL,
    CONFIG_URL,
    FALLBACK_UI_CONFIG,
    UI_CONFIG_STAGING_URL,
    UI_CONFIG_URL,
)
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)


def _is_ci_env_or_testing() -> bool:
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
        "TEAMCITY_VERSION",  # https://www.jetbrains.com/help/teamcity/predefined-build-parameters.html#Predefined+Server+Build+Parameters
        "CLOUD_RUN_EXECUTION",  # https://cloud.google.com/run/docs/reference/container-contract#env-vars
        # runbots
        "RUNBOT_HOST_URL",
        "RUNBOT_BUILD_NAME",
        "RUNBOT_WORKER_ID",
    }
    # For most of the cases, the env var existing means we are in CI
    for var in env_vars:
        if var in os.environ:
            return True
    return False


# NB: implement the function here to avoid unnecessary imports inside databricks_utils
def _is_in_databricks() -> bool:
    # check if in databricks runtime
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        return True
    if os.path.exists("/databricks/DBR_VERSION"):
        return True

    # check if in databricks model serving environment
    if os.environ.get("IS_IN_DB_MODEL_SERVING_ENV", "false").lower() == "true":
        return True

    return False


_IS_MLFLOW_DEV_VERSION = Version(VERSION).is_devrelease
_IS_IN_CI_ENV_OR_TESTING = _is_ci_env_or_testing()
_IS_IN_DATABRICKS = _is_in_databricks()
_IS_MLFLOW_TESTING_TELEMETRY = _MLFLOW_TESTING_TELEMETRY.get()


def is_telemetry_disabled() -> bool:
    try:
        if _IS_MLFLOW_TESTING_TELEMETRY:
            return False
        return (
            MLFLOW_DISABLE_TELEMETRY.get()
            or os.environ.get("DO_NOT_TRACK", "false").lower() == "true"
            or _IS_IN_CI_ENV_OR_TESTING
            or _IS_IN_DATABRICKS
            or _IS_MLFLOW_DEV_VERSION
        )
    except Exception as e:
        _log_error(f"Failed to check telemetry disabled status: {e}")
        return True


def _get_config_url(version: str, is_ui: bool = False) -> str | None:
    """
    Get the config URL for the given MLflow version.
    """
    version_obj = Version(version)

    if version_obj.is_devrelease or _IS_MLFLOW_TESTING_TELEMETRY:
        base_url = UI_CONFIG_STAGING_URL if is_ui else CONFIG_STAGING_URL
        return f"{base_url}/{version}.json"

    if version_obj.base_version == version or (
        version_obj.is_prerelease and version_obj.pre[0] == "rc"
    ):
        base_url = UI_CONFIG_URL if is_ui else CONFIG_URL
        return f"{base_url}/{version}.json"

    return None


def _log_error(message: str) -> None:
    if _MLFLOW_TELEMETRY_LOGGING.get():
        _logger.error(message, exc_info=True)


def fetch_ui_telemetry_config() -> dict[str, Any]:
    # Check if telemetry is disabled
    if is_telemetry_disabled():
        return FALLBACK_UI_CONFIG

    # Get config URL
    config_url = _get_config_url(VERSION, is_ui=True)
    if not config_url:
        return FALLBACK_UI_CONFIG

    # Fetch config from remote URL
    try:
        response = requests.get(config_url, timeout=1)
        if response.status_code != 200:
            return FALLBACK_UI_CONFIG

        return response.json()
    except Exception as e:
        _log_error(f"Failed to fetch UI telemetry config: {e}")
        return FALLBACK_UI_CONFIG
