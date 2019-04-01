from click.testing import CliRunner
from mock import mock

from mlflow.cli import server, run


def test_server_uri_validation():
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        # SQLAlchemy expects postgresql:// not postgres://
        CliRunner().invoke(server, ["--backend-store-uri", "postgres://user:pwd@host:5432/mydb"])
        run_server_mock.assert_not_called()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        # Option 'default-artifact-root' is required in this case
        CliRunner().invoke(server, ["--backend-store-uri", "sqlite://"])
        run_server_mock.assert_not_called()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        # Shouldn't have access to the S3 bucket
        CliRunner().invoke(server, ["--default-artifact-root", "bad-scheme://afdf/dfd"])
        run_server_mock.assert_not_called()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        # Shouldn't have access to the S3 bucket
        CliRunner().invoke(server, ["--default-artifact-root", "s3://private-bucket"])
        run_server_mock.assert_not_called()
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        # Requires the dependency google-cloud-storage
        CliRunner().invoke(server, ["--default-artifact-root", "gs://private-bucket"])
        run_server_mock.assert_not_called()


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


def test_mlflow_run():
    with mock.patch("mlflow.cli.projects") as mock_projects:
        result = CliRunner().invoke(run)
        mock_projects.run.assert_not_called()
        assert 'Missing argument "URI"' in result.output

    with mock.patch("mlflow.cli.projects") as mock_projects:
        CliRunner().invoke(run, ["project_uri"])
        mock_projects.run.assert_called_once()

    with mock.patch("mlflow.cli.projects") as mock_projects:
        CliRunner().invoke(run, ["--experiment-id", "5", "project_uri"])
        mock_projects.run.assert_called_once()

    with mock.patch("mlflow.cli.projects") as mock_projects:
        CliRunner().invoke(run, ["--experiment-name", "random name", "project_uri"])
        mock_projects.run.assert_called_once()

    with mock.patch("mlflow.cli.projects") as mock_projects:
        result = CliRunner().invoke(run, ["--experiment-id", "51",
                                          "--experiment-name", "name blah", "uri"])
        mock_projects.run.assert_not_called()
        assert "Specify only one of 'experiment-name' or 'experiment-id' options." in result.output
