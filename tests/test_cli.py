from click.testing import CliRunner
from mock import mock
import numpy as np
import os
import pandas as pd
import pytest
import shutil
import tempfile
import textwrap

from mlflow.cli import run, server, ui
from mlflow.server import handlers
from mlflow import experiments


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


def test_server_default_artifact_root_validation():
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        result = CliRunner().invoke(server, ["--backend-store-uri", "sqlite:///my.db"])
        assert result.output.startswith("Option 'default-artifact-root' is required")
        run_server_mock.assert_not_called()


@pytest.mark.parametrize("command", [server, ui])
def test_tracking_uri_validation_failure(command):
    handlers._store = None
    with mock.patch("mlflow.cli._run_server") as run_server_mock:
        # SQLAlchemy expects postgresql:// not postgres://
        CliRunner().invoke(command,
                           ["--backend-store-uri", "postgres://user:pwd@host:5432/mydb",
                            "--default-artifact-root", "./mlruns"])
        run_server_mock.assert_not_called()


@pytest.mark.parametrize("command", [server, ui])
def test_tracking_uri_validation_sql_driver_uris(command):
    handlers._store = None
    with mock.patch("mlflow.cli._run_server") as run_server_mock,\
            mock.patch("mlflow.store.sqlalchemy_store.SqlAlchemyStore") as sql_store:
        CliRunner().invoke(command,
                           ["--backend-store-uri", "mysql+pymysql://user:pwd@host:5432/mydb",
                            "--default-artifact-root", "./mlruns"])
        sql_store.assert_called_once_with("mysql+pymysql://user:pwd@host:5432/mydb", "./mlruns")
        run_server_mock.assert_called()


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


def test_csv_generation():
    with mock.patch('mlflow.experiments.fluent.search_runs') as mock_search_runs:
        mock_search_runs.return_value = pd.DataFrame({
            "run_id": np.array(["all_set", "with_none", "with_nan"]),
            "experiment_id": np.array([1, 1, 1]),
            "param_optimizer": np.array(["Adam", None, "Adam"]),
            "avg_loss": np.array([42.0, None, np.nan], dtype=np.float32)},
            columns=["run_id", "experiment_id", "param_optimizer", "avg_loss"])
        expected_csv = textwrap.dedent("""\
        run_id,experiment_id,param_optimizer,avg_loss
        all_set,1,Adam,42.0
        with_none,1,,
        with_nan,1,Adam,
        """)
        tempdir = tempfile.mkdtemp()
        try:
            result_filename = os.path.join(tempdir, "result.csv")
            CliRunner().invoke(experiments.generate_csv_with_runs,
                               ["--experiment-id", "1",
                                "--filename", result_filename])
            with open(result_filename, 'r') as fd:
                assert expected_csv == fd.read()
        finally:
            shutil.rmtree(tempdir)
