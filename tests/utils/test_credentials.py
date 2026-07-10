from unittest import mock
from unittest.mock import patch

import pytest

from mlflow import get_tracking_uri
from mlflow.environment_variables import (
    MLFLOW_ENTRA_ID_SCOPE,
    MLFLOW_TRACKING_AUTH,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_URI,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.exceptions import MlflowException
from mlflow.utils.credentials import login, read_mlflow_creds


def test_read_mlflow_creds_file(tmp_path, monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)

    creds_file = tmp_path.joinpath("credentials")
    with mock.patch("mlflow.utils.credentials._get_credentials_path", return_value=str(creds_file)):
        # credentials file does not exist
        creds = read_mlflow_creds()
        assert creds.username is None
        assert creds.password is None

        # credentials file is empty
        creds = read_mlflow_creds()
        assert creds.username is None
        assert creds.password is None

        # password is missing
        creds_file.write_text(
            """
[mlflow]
mlflow_tracking_username = username
"""
        )
        creds = read_mlflow_creds()
        assert creds.username == "username"
        assert creds.password is None

        # username is missing
        creds_file.write_text(
            """
[mlflow]
mlflow_tracking_password = password
"""
        )
        creds = read_mlflow_creds()
        assert creds.username is None
        assert creds.password == "password"

        # valid credentials
        creds_file.write_text(
            """
[mlflow]
mlflow_tracking_username = username
mlflow_tracking_password = password
"""
        )
        creds = read_mlflow_creds()
        assert creds is not None
        assert creds.username == "username"
        assert creds.password == "password"


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("username", "password"),
        ("username", None),
        (None, "password"),
        (None, None),
    ],
)
def test_read_mlflow_creds_env(username, password, monkeypatch):
    if username is None:
        monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    else:
        monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, username)

    if password is None:
        monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)
    else:
        monkeypatch.setenv(MLFLOW_TRACKING_PASSWORD.name, password)

    creds = read_mlflow_creds()
    assert creds.username == username
    assert creds.password == password


def test_read_mlflow_creds_env_takes_precedence_over_file(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, "username_env")
    monkeypatch.setenv(MLFLOW_TRACKING_PASSWORD.name, "password_env")
    creds_file = tmp_path.joinpath("credentials")
    with mock.patch("mlflow.utils.credentials._get_credentials_path", return_value=str(creds_file)):
        creds_file.write_text(
            """
[mlflow]
mlflow_tracking_username = username_file
mlflow_tracking_password = password_file
"""
        )
        creds = read_mlflow_creds()
        assert creds.username == "username_env"
        assert creds.password == "password_env"


def test_mlflow_login(tmp_path, monkeypatch):
    # Mock `input()` and `getpass()` to return host, username and password in order.
    with (
        patch(
            "builtins.input",
            side_effect=["https://community.cloud.databricks.com/", "dummyusername"],
        ),
        patch("getpass.getpass", side_effect=["dummypassword"]),
    ):
        file_name = f"{tmp_path}/.databrickscfg"
        profile = "TEST"
        monkeypatch.setenv("DATABRICKS_CONFIG_FILE", file_name)
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", profile)

        def success():
            return

        with patch(
            "mlflow.utils.credentials._validate_databricks_auth",
            side_effect=[MlflowException("Invalid databricks credentials."), success()],
        ):
            login("databricks")

    with open(file_name) as f:
        lines = f.readlines()
        assert lines[0] == "[TEST]\n"
        assert lines[1] == "host = https://community.cloud.databricks.com/\n"
        assert lines[2] == "username = dummyusername\n"
        assert lines[3] == "password = dummypassword\n"

    # Assert that the tracking URI is set to the databricks.
    assert get_tracking_uri() == "databricks"


def test_mlflow_login_noninteractive():
    # Forces mlflow.utils.credentials._validate_databricks_auth to raise `MlflowException()`
    with patch(
        "mlflow.utils.credentials._validate_databricks_auth",
        side_effect=MlflowException("Failed to validate databricks credentials."),
    ):
        with pytest.raises(
            MlflowException,
            match="No valid Databricks credentials found while running in non-interactive mode",
        ):
            login(backend="databricks", interactive=False)


def _clear_env_var(monkeypatch, env_var):
    # `monkeypatch.setenv` records the original state (including absence) so that
    # values set by the code under test are reverted after the test.
    monkeypatch.setenv(env_var.name, "placeholder")
    monkeypatch.delenv(env_var.name)


def test_mlflow_login_entra(monkeypatch):
    for env_var in (MLFLOW_ENTRA_ID_SCOPE, MLFLOW_TRACKING_AUTH, MLFLOW_TRACKING_URI):
        _clear_env_var(monkeypatch, env_var)

    # Mock `input()` to return the token scope and host in order.
    with (
        patch(
            "builtins.input",
            side_effect=["api://dummy-client-id/.default", "https://mlflow.example.com"],
        ),
        patch("mlflow.utils.credentials._validate_entra_auth") as validate_mock,
    ):
        login(backend="entra")

    validate_mock.assert_called_once()
    assert MLFLOW_ENTRA_ID_SCOPE.get() == "api://dummy-client-id/.default"
    assert MLFLOW_TRACKING_AUTH.get() == "entra"
    assert get_tracking_uri() == "https://mlflow.example.com"


def test_mlflow_login_entra_from_env(monkeypatch):
    _clear_env_var(monkeypatch, MLFLOW_TRACKING_AUTH)
    monkeypatch.setenv(MLFLOW_ENTRA_ID_SCOPE.name, "api://dummy-client-id/.default")
    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, "https://mlflow.example.com")

    with patch("mlflow.utils.credentials._validate_entra_auth") as validate_mock:
        login(backend="entra", interactive=False)

    validate_mock.assert_called_once()
    assert MLFLOW_TRACKING_AUTH.get() == "entra"
    assert get_tracking_uri() == "https://mlflow.example.com"


def test_mlflow_login_entra_noninteractive_missing_scope(monkeypatch):
    for env_var in (MLFLOW_ENTRA_ID_SCOPE, MLFLOW_TRACKING_AUTH, MLFLOW_TRACKING_URI):
        _clear_env_var(monkeypatch, env_var)

    with pytest.raises(
        MlflowException,
        match=f"{MLFLOW_ENTRA_ID_SCOPE.name} must be set while running in non-interactive mode",
    ):
        login(backend="entra", interactive=False)


def test_mlflow_login_entra_noninteractive_missing_uri(monkeypatch):
    for env_var in (MLFLOW_TRACKING_AUTH, MLFLOW_TRACKING_URI):
        _clear_env_var(monkeypatch, env_var)
    monkeypatch.setenv(MLFLOW_ENTRA_ID_SCOPE.name, "api://dummy-client-id/.default")

    with pytest.raises(
        MlflowException,
        match=f"{MLFLOW_TRACKING_URI.name} must be set to an HTTP\\(S\\) tracking server URI",
    ):
        login(backend="entra", interactive=False)


def test_mlflow_login_entra_invalid_credentials(monkeypatch):
    _clear_env_var(monkeypatch, MLFLOW_TRACKING_AUTH)
    monkeypatch.setenv(MLFLOW_ENTRA_ID_SCOPE.name, "api://dummy-client-id/.default")
    monkeypatch.setenv(MLFLOW_TRACKING_URI.name, "https://mlflow.example.com")

    with patch(
        "mlflow.utils.credentials._validate_entra_auth",
        side_effect=MlflowException("Failed to validate Microsoft Entra ID credentials."),
    ):
        with pytest.raises(MlflowException, match="`mlflow.login\\(\\)` failed with error"):
            login(backend="entra", interactive=False)

    assert MLFLOW_TRACKING_AUTH.get() is None


def test_mlflow_login_unsupported_backend():
    with pytest.raises(MlflowException, match="received `backend=foo`"):
        login(backend="foo")
