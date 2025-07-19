import os
import random
import re
from typing import Optional

import requests

from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY
from mlflow.telemetry.constant import BASE_URL
from mlflow.telemetry.schemas import TelemetryConfig, get_source_sdk
from mlflow.version import VERSION


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
    if (
        os.environ.get("IS_IN_DB_MODEL_SERVING_ENV")
        or os.environ.get("IS_IN_DATABRICKS_MODEL_SERVING_ENV")
        or "false"
    ).lower() == "true":
        return True

    return False


_IS_MLFLOW_DEV_VERSION = VERSION.endswith(".dev0")
_IS_IN_CI_ENV_OR_TESTING = _is_ci_env_or_testing()
_IS_IN_DATABRICKS = _is_in_databricks()


def is_telemetry_disabled() -> bool:
    return (
        MLFLOW_DISABLE_TELEMETRY.get()
        or os.environ.get("DO_NOT_TRACK", "false").lower() == "true"
        or _IS_IN_CI_ENV_OR_TESTING
        or _IS_IN_DATABRICKS
        or _IS_MLFLOW_DEV_VERSION
    )


def _get_config_url(version: str) -> Optional[str]:
    """
    Get the config URL for the given MLflow version.
    """
    pattern = r"^(\d+)\.(\d+)\.(\d+)(\.rc\d+)?$"

    if re.match(pattern, version):
        return f"{BASE_URL}/config/{version}"

    if version.endswith(".dev0"):
        return f"{BASE_URL}/dev/config/{version}"

    return None


def _get_config() -> Optional[TelemetryConfig]:
    """
    Get the config for the given MLflow version.
    """
    if config_url := _get_config_url(VERSION):
        try:
            response = requests.get(config_url, timeout=1)
            if response.status_code != 200:
                return None
            config = response.json()
            if (
                config.get("mlflow_version") != VERSION
                or config.get("disable_telemetry") is True
                or config.get("telemetry_url") is None
            ):
                return None

            rollout_percentage = config.get("rollout_percentage", 100)
            if random.randint(0, 100) > rollout_percentage:
                return None

            if get_source_sdk().value in config.get("disable_sdks", []):
                return None

            return TelemetryConfig(
                telemetry_url=config["telemetry_url"],
                disable_api_map=config.get("disable_api_map", {}),
            )
        except Exception:
            return None

    return None
