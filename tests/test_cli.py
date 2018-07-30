from click.testing import CliRunner
from mock import mock

from mlflow.cli import server


def test_server_static_prefix_validation():
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        CliRunner().invoke(server)
        run_server_mock.assert_called_once()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        CliRunner().invoke(server, ["--static-prefix", "/mlflow"])
        run_server_mock.assert_called_once()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        result = CliRunner().invoke(server, ["--static-prefix", "mlflow/"])
        assert "--static-prefix must begin with a '/'." in result.output
        run_server_mock.assert_not_called()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        result = CliRunner().invoke(server, ["--static-prefix", "/mlflow/"])
        assert "--static-prefix should not end with a '/'." in result.output
        run_server_mock.assert_not_called()
