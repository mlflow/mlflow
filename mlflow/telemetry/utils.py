import logging
import os
import re
from pathlib import Path
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
from mlflow.telemetry.schemas import ENV_VAR_TO_ENVIRONMENT_MAP, CodingAgent, Environment
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)

_AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


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
    if os.environ.get("IS_IN_DB_MODEL_SERVING_ENV", "false").lower() in ("true", "1"):
        return True

    return False


def _detect_environment() -> str | None:
    # Check for MLflow demo deployment (e.g. demo.mlflow.org) before generic docker detection
    if os.environ.get("MLFLOW_DEPLOYMENT_ENV") == "demo":
        return Environment.DEMO.value

    for env_var, environment in ENV_VAR_TO_ENVIRONMENT_MAP.items():
        if env_var in os.environ:
            return environment.value

    # https://docs.aws.amazon.com/sagemaker/latest/dg/nbi-metadata.html
    if Path("/opt/ml/metadata/resource-metadata.json").exists():
        return Environment.SAGEMAKER_NOTEBOOK.value
    # unofficial heuristic to detect docker environment
    if Path("/.dockerenv").exists():
        return Environment.DOCKER.value
    return None


def _detect_agent() -> str | None:
    """Return the coding agent driving this process, if one can be identified."""
    # https://www.npmjs.com/package/std-env#agent-detection
    if agent := os.environ.get("AI_AGENT", "").strip():
        if _AGENT_NAME_PATTERN.fullmatch(agent):
            agent = agent.lower()
            for known in sorted(CodingAgent, key=lambda a: len(a.value), reverse=True):
                if agent == known.value or (
                    agent.startswith(f"{known.value}_") and agent.endswith("_agent")
                ):
                    return known.value
            return agent

    def is_set(*names: str) -> bool:
        return any(os.environ.get(name) for name in names)

    # Amp also sets Claude Code markers, but it is not included in telemetry detection.
    if os.environ.get("AGENT") == "amp" or is_set("AMP_CURRENT_THREAD_ID"):
        return None
    if is_set("CODEX_SANDBOX", "CODEX_CI", "CODEX_THREAD_ID"):
        return CodingAgent.CODEX.value
    if is_set("GEMINI_CLI"):
        return CodingAgent.GEMINI_CLI.value
    if is_set("COPILOT_CLI"):
        return CodingAgent.COPILOT_CLI.value
    if is_set("OPENCODE"):
        return CodingAgent.OPENCODE.value
    if is_set("CLINE_ACTIVE", "CLINE_AGENT"):
        return CodingAgent.CLINE.value
    if is_set("CLAUDECODE", "CLAUDE_CODE"):
        if is_set("CLAUDE_CODE_IS_COWORK"):
            return None
        return CodingAgent.CLAUDE_CODE.value
    if is_set("CURSOR_TRACE_ID"):
        return CodingAgent.CURSOR.value
    if is_set("CURSOR_AGENT") or os.environ.get("CURSOR_EXTENSION_HOST_ROLE") == "agent-exec":
        return CodingAgent.CURSOR_CLI.value
    if is_set("KIRO_AGENT_PATH"):
        return CodingAgent.KIRO.value
    if is_set("PI_CODING_AGENT"):
        return CodingAgent.PI.value
    return None


_IS_MLFLOW_DEV_VERSION = Version(VERSION).is_devrelease
_IS_IN_CI_ENV_OR_TESTING = _is_ci_env_or_testing()
_IS_IN_DATABRICKS = _is_in_databricks()
_IS_MLFLOW_TESTING_TELEMETRY = _MLFLOW_TESTING_TELEMETRY.get()


def is_telemetry_disabled() -> bool:
    try:
        if _IS_MLFLOW_TESTING_TELEMETRY:
            return False
        # NB: _IS_IN_DATABRICKS is intentionally NOT a disable signal here. When the
        # tracking URI is databricks:// or databricks-uc://, telemetry is forwarded
        # to the workspace's own ingestion endpoint (see TelemetryClient._forward_to_databricks).
        # The non-Databricks (OSS) ingestion path is separately guarded in
        # TelemetryClient._process_records so it is never hit from inside DBR.
        return (
            MLFLOW_DISABLE_TELEMETRY.get()
            or os.environ.get("DO_NOT_TRACK", "false").lower() == "true"
            or _IS_IN_CI_ENV_OR_TESTING
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
