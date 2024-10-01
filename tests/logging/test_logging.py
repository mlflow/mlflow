import logging
import os

import pytest

from mlflow.utils.logging_utils import set_mlflow_log_level


def test_mlflow_log_level_environment_variable():
    # Test default level
    set_mlflow_log_level()
    assert logging.getLogger("mlflow").level == logging.INFO

    # Test setting to DEBUG
    os.environ["MLFLOW_LOGGING_LEVEL"] = "DEBUG"
    set_mlflow_log_level()
    assert logging.getLogger("mlflow").level == logging.DEBUG

    # Test invalid level (should raise ValueError with a specific message)
    os.environ["MLFLOW_LOGGING_LEVEL"] = "INVALID_LEVEL"
    with pytest.raises(ValueError, match="Invalid log level: INVALID_LEVEL"):
        set_mlflow_log_level()

    # Clean up
    del os.environ["MLFLOW_LOGGING_LEVEL"]
