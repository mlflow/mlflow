from unittest import mock

import pytest

from mlflow.environment_variables import MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_USERNAME
from mlflow.utils.credentials import read_mlflow_creds


def test_read_mlflow_creds_file(tmp_path, monkeypatch):
    monkeypatch.delenvs(
        (MLFLOW_TRACKING_USERNAME.name, MLFLOW_TRACKING_PASSWORD.name), raising=False
    )

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
