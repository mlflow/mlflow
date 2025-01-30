import logging

import pytest

from mlflow.environment_variables import MLFLOW_LOGGING_LEVEL
from mlflow.utils.logging_utils import set_mlflow_log_level, _configure_mlflow_loggers


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure MLflow logger before each test"""
    _configure_mlflow_loggers("mlflow")
    yield
    # Clean up after test
    MLFLOW_LOGGING_LEVEL.unset()


def test_mlflow_log_level_environment_variable():
    # Test default level
    set_mlflow_log_level()
    assert logging.getLogger("mlflow").level == logging.INFO

    # Test setting to DEBUG
    MLFLOW_LOGGING_LEVEL.set("DEBUG")
    set_mlflow_log_level()
    assert logging.getLogger("mlflow").level == logging.DEBUG

    # Test invalid level (should raise ValueError)
    MLFLOW_LOGGING_LEVEL.set("INVALID_LEVEL")
    with pytest.raises(ValueError, match="Invalid log level: INVALID_LEVEL"):
        set_mlflow_log_level()
