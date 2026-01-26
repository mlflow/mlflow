"""Utilities for environment variable validation and MLflow configuration."""

import os

from packaging import version


def get_env_vars() -> dict[str, str | None]:
    """Get MLflow environment variables.

    Returns:
        Dictionary with tracking_uri and experiment_id (may be None)
    """
    return {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
        "experiment_id": os.getenv("MLFLOW_EXPERIMENT_ID"),
    }


def validate_env_vars(
    require_tracking_uri: bool = True, require_experiment_id: bool = True
) -> list[str]:
    """Validate required environment variables are set.

    Args:
        require_tracking_uri: If True, MLFLOW_TRACKING_URI must be set
        require_experiment_id: If True, MLFLOW_EXPERIMENT_ID must be set

    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    env_vars = get_env_vars()

    if require_tracking_uri and not env_vars["tracking_uri"]:
        errors.append("MLFLOW_TRACKING_URI is not set")

    if require_experiment_id and not env_vars["experiment_id"]:
        errors.append("MLFLOW_EXPERIMENT_ID is not set")

    return errors


def validate_mlflow_version(min_version: str = "3.8.0") -> tuple[bool, str]:
    """Check MLflow version compatibility.

    Args:
        min_version: Minimum required MLflow version

    Returns:
        Tuple of (is_valid, version_string)
    """
    try:
        import mlflow

        current_version = mlflow.__version__

        # Remove dev/rc suffixes for comparison
        clean_version = current_version.split("dev")[0].split("rc")[0]

        is_valid = version.parse(clean_version) >= version.parse(min_version)
        return is_valid, current_version
    except ImportError:
        return False, "not installed"


def test_mlflow_connection(tracking_uri: str, experiment_id: str) -> tuple[bool, str]:
    """Test connection to MLflow tracking server.

    Args:
        tracking_uri: MLflow tracking URI
        experiment_id: MLflow experiment ID

    Returns:
        Tuple of (success, error_message_or_experiment_name)
    """
    try:
        from mlflow import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment(experiment_id)

        if experiment:
            return True, experiment.name
        else:
            return False, f"Experiment {experiment_id} not found"
    except Exception as e:
        return False, str(e)[:100]


def check_databricks_config() -> tuple[bool, str | None]:
    """Check if running with Databricks configuration.

    Returns:
        Tuple of (is_databricks, profile_or_error_message)
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")

    # Check if tracking URI indicates Databricks
    if "databricks" in tracking_uri.lower():
        # Extract profile if present
        if "databricks://" in tracking_uri:
            profile = tracking_uri.split("databricks://")[1] if len(tracking_uri.split("databricks://")) > 1 else "DEFAULT"
            return True, profile
        return True, "databricks"

    # Check for Databricks SDK/CLI
    try:
        import subprocess

        result = subprocess.run(
            ["databricks", "auth", "profiles"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            profiles = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            return True, profiles[0] if profiles else None
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        pass

    return False, None
