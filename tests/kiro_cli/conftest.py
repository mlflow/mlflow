import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    """Provide a Click CliRunner instance for CLI tests."""
    return CliRunner()


@pytest.fixture
def tmp_settings_path(tmp_path):
    """Provide a temporary settings.json path for tests."""
    return tmp_path / "settings.json"


@pytest.fixture
def sample_env_dict():
    """Provide a sample env dict with MLflow keys for reuse across tests."""
    return {
        "MLFLOW_KIRO_CLI_TRACING_ENABLED": "true",
        "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db",
        "MLFLOW_EXPERIMENT_ID": "123",
    }
