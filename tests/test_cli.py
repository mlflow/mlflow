from click.testing import CliRunner
from unittest import mock
import numpy as np
import os
import pandas as pd
import pytest
import shutil
import tempfile
import textwrap
import time
import subprocess

from urllib.request import url2pathname
from urllib.parse import urlparse, unquote

from mlflow.cli import run, server
from mlflow import experiments
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.file_store import FileStore
from mlflow.exceptions import MlflowException
from mlflow.entities import ViewType


def test_mlflow_run():
    with mock.patch("mlflow.cli.projects") as mock_projects:
        result = CliRunner().invoke(run)
        mock_projects.run.assert_not_called()
        assert "Missing argument 'URI'" in result.output

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
        result = CliRunner().invoke(
            run, ["--experiment-id", "51", "--experiment-name", "name blah", "uri"]
        )
        mock_projects.run.assert_not_called()
        assert "Specify only one of 'experiment-name' or 'experiment-id' options." in result.output


def test_csv_generation():
    with mock.patch("mlflow.experiments.fluent.search_runs") as mock_search_runs:
        mock_search_runs.return_value = pd.DataFrame(
            {
                "run_id": np.array(["all_set", "with_none", "with_nan"]),
                "experiment_id": np.array([1, 1, 1]),
                "param_optimizer": np.array(["Adam", None, "Adam"]),
                "avg_loss": np.array([42.0, None, np.nan], dtype=np.float32),
            },
            columns=["run_id", "experiment_id", "param_optimizer", "avg_loss"],
        )
        expected_csv = textwrap.dedent(
            """\
        run_id,experiment_id,param_optimizer,avg_loss
        all_set,1,Adam,42.0
        with_none,1,,
        with_nan,1,Adam,
        """
        )
        tempdir = tempfile.mkdtemp()
        try:
            result_filename = os.path.join(tempdir, "result.csv")
            CliRunner().invoke(
                experiments.generate_csv_with_runs,
                ["--experiment-id", "1", "--filename", result_filename],
            )
            with open(result_filename, "r") as fd:
                assert expected_csv == fd.read()
        finally:
            shutil.rmtree(tempdir)


@pytest.fixture(scope="function")
def sqlite_store():
    fd, temp_dbfile = tempfile.mkstemp()
    # Close handle immediately so that we can remove the file later on in Windows
    os.close(fd)
    db_uri = "sqlite:///%s" % temp_dbfile
    store = SqlAlchemyStore(db_uri, "artifact_folder")
    yield (store, db_uri)
    os.remove(temp_dbfile)
    shutil.rmtree("artifact_folder")


@pytest.fixture(scope="function")
def file_store():
    ROOT_LOCATION = os.path.join(tempfile.gettempdir(), "test_mlflow_gc")
    file_store_uri = "file:///%s" % ROOT_LOCATION
    yield (FileStore(ROOT_LOCATION), file_store_uri)
    shutil.rmtree(ROOT_LOCATION)


def _create_run_in_store(store):
    config = {
        "experiment_id": "0",
        "user_id": "Anderson",
        "start_time": int(time.time()),
        "tags": {},
    }
    run = store.create_run(**config)
    artifact_path = url2pathname(unquote(urlparse(run.info.artifact_uri).path))
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path)
    return run


def test_mlflow_gc_sqlite(sqlite_store):
    store = sqlite_store[0]
    run = _create_run_in_store(store)
    store.delete_run(run.info.run_uuid)
    subprocess.check_output(["mlflow", "gc", "--backend-store-uri", sqlite_store[1]])
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 0
    with pytest.raises(MlflowException):
        store.get_run(run.info.run_uuid)


def test_mlflow_gc_file_store(file_store):
    store = file_store[0]
    run = _create_run_in_store(store)
    store.delete_run(run.info.run_uuid)
    subprocess.check_output(["mlflow", "gc", "--backend-store-uri", file_store[1]])
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 0
    with pytest.raises(MlflowException):
        store.get_run(run.info.run_uuid)


def test_mlflow_gc_file_store_passing_explicit_run_ids(file_store):
    store = file_store[0]
    run = _create_run_in_store(store)
    store.delete_run(run.info.run_uuid)
    subprocess.check_output(
        ["mlflow", "gc", "--backend-store-uri", file_store[1], "--run-ids", run.info.run_uuid]
    )
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 0
    with pytest.raises(MlflowException):
        store.get_run(run.info.run_uuid)


def test_mlflow_gc_not_deleted_run(file_store):
    store = file_store[0]
    run = _create_run_in_store(store)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_output(
            ["mlflow", "gc", "--backend-store-uri", file_store[1], "--run-ids", run.info.run_uuid]
        )
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 1
