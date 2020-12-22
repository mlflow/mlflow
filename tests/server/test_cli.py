from click.testing import CliRunner
from unittest import mock
import pytest

from mlflow.cli import server, ui
from mlflow.server import handlers


def test_server_static_prefix_validation():
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        CliRunner().invoke(server)
        run_server_mock.assert_called_once()
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        CliRunner().invoke(server, ["--static-prefix", "/mlflow"])
        run_server_mock.assert_called_once()
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        result = CliRunner().invoke(server, ["--static-prefix", "mlflow/"])
        assert "--static-prefix must begin with a '/'." in result.output
        run_server_mock.assert_not_called()
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        result = CliRunner().invoke(server, ["--static-prefix", "/mlflow/"])
        assert "--static-prefix should not end with a '/'." in result.output
        run_server_mock.assert_not_called()


def test_server_default_artifact_root_validation():
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        result = CliRunner().invoke(server, ["--backend-store-uri", "sqlite:///my.db"])
        assert result.output.startswith("Option 'default-artifact-root' is required")
        run_server_mock.assert_not_called()


@pytest.mark.parametrize("command", [server, ui])
def test_tracking_uri_validation_failure(command):
    handlers._tracking_store = None
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        # SQLAlchemy expects postgresql:// not postgres://
        CliRunner().invoke(
            command,
            [
                "--backend-store-uri",
                "postgres://user:pwd@host:5432/mydb",
                "--default-artifact-root",
                "./mlruns",
            ],
        )
        run_server_mock.assert_not_called()


@pytest.mark.parametrize("command", [server, ui])
def test_tracking_uri_validation_sql_driver_uris(command):
    handlers._tracking_store = None
    handlers._model_registry_store = None
    with mock.patch("mlflow.server._run_server") as run_server_mock, mock.patch(
        "mlflow.store.tracking.sqlalchemy_store.SqlAlchemyStore"
    ) as tracking_store_mock, mock.patch(
        "mlflow.store.model_registry.sqlalchemy_store.SqlAlchemyStore"
    ) as registry_store_mock:
        CliRunner().invoke(
            command,
            [
                "--backend-store-uri",
                "mysql+pymysql://user:pwd@host:5432/mydb",
                "--default-artifact-root",
                "./mlruns",
            ],
        )
        tracking_store_mock.assert_called_once_with(
            "mysql+pymysql://user:pwd@host:5432/mydb", "./mlruns"
        )
        registry_store_mock.assert_called_once_with("mysql+pymysql://user:pwd@host:5432/mydb")
        run_server_mock.assert_called()
