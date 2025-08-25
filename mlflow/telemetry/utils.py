import os

from packaging.version import Version

from mlflow.environment_variables import _MLFLOW_TESTING_TELEMETRY, MLFLOW_DISABLE_TELEMETRY
from mlflow.telemetry.constant import CONFIG_STAGING_URL, CONFIG_URL
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


_IS_MLFLOW_DEV_VERSION = Version(VERSION).is_devrelease
_IS_IN_CI_ENV_OR_TESTING = _is_ci_env_or_testing()
_IS_IN_DATABRICKS = _is_in_databricks()
_IS_MLFLOW_TESTING = _MLFLOW_TESTING_TELEMETRY.get()


def is_telemetry_disabled() -> bool:
    try:
        if _IS_MLFLOW_TESTING:
            return False
        return (
            MLFLOW_DISABLE_TELEMETRY.get()
            or os.environ.get("DO_NOT_TRACK", "false").lower() == "true"
            or _IS_IN_CI_ENV_OR_TESTING
            or _IS_IN_DATABRICKS
            or _IS_MLFLOW_DEV_VERSION
        )
    except Exception:
        return True


def _get_config_url(version: str) -> str | None:
    """
    Get the config URL for the given MLflow version.
    """
    version_obj = Version(version)

    if version_obj.is_devrelease or _IS_MLFLOW_TESTING:
        return f"{CONFIG_STAGING_URL}/{version}.json"

    if version_obj.base_version == version or (
        version_obj.is_prerelease and version_obj.pre[0] == "rc"
    ):
        return f"{CONFIG_URL}/{version}.json"

    return None
