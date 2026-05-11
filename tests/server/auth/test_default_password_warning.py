import logging

import pytest

from mlflow.server.auth import _logger as auth_logger
from mlflow.server.auth import _warn_if_default_admin_password


@pytest.fixture
def auth_caplog(caplog):
    # The "mlflow" logger is configured with propagate=False (see
    # mlflow.utils.logging_utils._configure_mlflow_loggers), so caplog's default
    # root-handler capture misses records from mlflow.server.auth. Attach
    # caplog's handler directly to the auth logger for the duration of the test.
    auth_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        auth_logger.removeHandler(caplog.handler)


@pytest.mark.parametrize(
    "password",
    [
        "not-the-default",
        "another-strong-password",
        "",
        "password",
    ],
)
def test_no_warning_when_admin_password_is_not_default(auth_caplog, password):
    with auth_caplog.at_level(logging.WARNING, logger="mlflow.server.auth"):
        _warn_if_default_admin_password(password)
    assert auth_caplog.records == []


def test_warning_emitted_when_admin_password_is_default(auth_caplog):
    with auth_caplog.at_level(logging.WARNING, logger="mlflow.server.auth"):
        _warn_if_default_admin_password("password1234")
    assert len(auth_caplog.records) == 1
    record = auth_caplog.records[0]
    assert record.levelname == "WARNING"
    assert "default password" in record.message
    assert "MLFLOW_AUTH_CONFIG_PATH" in record.message
