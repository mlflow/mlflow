from click.testing import CliRunner
from unittest import mock
import json
import os
import pytest
import shutil
import tempfile
import time
import subprocess
import requests

from urllib.request import url2pathname
from urllib.parse import urlparse, unquote
import numpy as np
import pandas as pd

import mlflow
from mlflow.cli import server, ui
from mlflow import pyfunc
from mlflow.server import handlers
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.file_store import FileStore
from mlflow.exceptions import MlflowException
from mlflow.entities import ViewType
from mlflow.utils.rest_utils import augmented_raise_for_status

from tests.helper_functions import pyfunc_serve_and_score_model, get_safe_port
from tests.tracking.integration_test_utils import _await_server_up_or_die


@pytest.mark.parametrize("command", ["server", "ui"])
def test_mlflow_server_command(command):
    port = get_safe_port()
    cmd = ["mlflow", command, "--port", str(port)]
    process = subprocess.Popen(cmd)
    try:
        _await_server_up_or_die(port, timeout=10)
        resp = requests.get(f"http://localhost:{port}/health")
        augmented_raise_for_status(resp)
        assert resp.text == "OK"
    finally:
        process.kill()


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


def test_server_mlflow_artifacts_options():
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        CliRunner().invoke(server, ["--artifacts-only"])
        run_server_mock.assert_called_once()
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        CliRunner().invoke(server, ["--serve-artifacts"])
        run_server_mock.assert_called_once()
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        CliRunner().invoke(server, ["--artifacts-only", "--serve-artifacts"])
        run_server_mock.assert_called_once()


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
    ), mock.patch("mlflow.store.model_registry.sqlalchemy_store.SqlAlchemyStore"):
        result = CliRunner().invoke(
            command,
            [
                "--backend-store-uri",
                "mysql+pymysql://user:pwd@host:5432/mydb",
                "--default-artifact-root",
                "./mlruns",
            ],
        )
        assert result.exit_code == 0
        run_server_mock.assert_called()


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
    with pytest.raises(MlflowException, match=r"Run .+ not found"):
        store.get_run(run.info.run_uuid)


def test_mlflow_gc_file_store(file_store):
    store = file_store[0]
    run = _create_run_in_store(store)
    store.delete_run(run.info.run_uuid)
    subprocess.check_output(["mlflow", "gc", "--backend-store-uri", file_store[1]])
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 0
    with pytest.raises(MlflowException, match=r"Run .+ not found"):
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
    with pytest.raises(MlflowException, match=r"Run .+ not found"):
        store.get_run(run.info.run_uuid)


def test_mlflow_gc_not_deleted_run(file_store):
    store = file_store[0]
    run = _create_run_in_store(store)
    with pytest.raises(subprocess.CalledProcessError, match=r".+"):
        subprocess.check_output(
            ["mlflow", "gc", "--backend-store-uri", file_store[1], "--run-ids", run.info.run_uuid]
        )
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 1


@pytest.mark.parametrize(
    "enable_mlserver",
    [
        # MLServer is not supported in Windows yet, so let's skip this test in that case.
        # https://github.com/SeldonIO/MLServer/issues/361
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                os.name == "nt", reason="MLServer is not supported in Windows"
            ),
        ),
        False,
    ],
)
def test_mlflow_models_serve(enable_mlserver):
    class MyModel(pyfunc.PythonModel):
        def predict(self, context, model_input):  # pylint: disable=unused-variable
            return np.array([1, 2, 3])

    model = MyModel()

    with mlflow.start_run():
        if enable_mlserver:
            # MLServer requires Python 3.7, so we'll force that Python version.
            with mock.patch("mlflow.utils.environment.PYTHON_VERSION", "3.7"):
                # We also need that MLServer is present on the Conda
                # environment, so we'll add that as an extra requirement.
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=model,
                    extra_pip_requirements=["mlserver", "mlserver-mlflow"],
                )
        else:
            mlflow.pyfunc.log_model(artifact_path="model", python_model=model)
        model_uri = mlflow.get_artifact_uri("model")

    data = pd.DataFrame({"a": [0]})

    extra_args = ["--no-conda"]
    if enable_mlserver:
        # When MLServer is enabled, we want to use Conda to ensure Python 3.7
        # is used
        extra_args = ["--enable-mlserver"]

    scoring_response = pyfunc_serve_and_score_model(
        model_uri=model_uri,
        data=data,
        content_type=pyfunc.scoring_server.CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        extra_args=extra_args,
    )
    assert scoring_response.status_code == 200
    served_model_preds = np.array(json.loads(scoring_response.content))
    np.testing.assert_array_equal(served_model_preds, model.predict(data, None))


def test_mlflow_tracking_disabled_in_artifacts_only_mode():

    port = get_safe_port()
    cmd = ["mlflow", "server", "--port", str(port), "--artifacts-only"]
    process = subprocess.Popen(cmd)
    _await_server_up_or_die(port, timeout=10)
    resp = requests.get(f"http://localhost:{port}/api/2.0/mlflow/experiments/list")
    assert (
        "Endpoint: /api/2.0/mlflow/experiments/list disabled due to the mlflow server running "
        "in `--artifacts-only` mode." in resp.text
    )
    process.kill()


def test_mlflow_artifact_list_in_artifacts_only_mode():

    port = get_safe_port()
    cmd = ["mlflow", "server", "--port", str(port), "--artifacts-only", "--serve-artifacts"]
    process = subprocess.Popen(cmd)
    try:
        _await_server_up_or_die(port, timeout=10)
        resp = requests.get(f"http://localhost:{port}/api/2.0/mlflow-artifacts/artifacts")
        augmented_raise_for_status(resp)
        assert resp.status_code == 200
        assert resp.text == "{}"
    finally:
        process.kill()


def test_mlflow_artifact_service_unavailable_without_config():

    port = get_safe_port()
    cmd = ["mlflow", "server", "--port", str(port)]
    process = subprocess.Popen(cmd)
    try:
        _await_server_up_or_die(port, timeout=10)
        endpoint = "/api/2.0/mlflow-artifacts/artifacts"
        resp = requests.get(f"http://localhost:{port}{endpoint}")
        assert (
            f"Endpoint: {endpoint} disabled due to the mlflow server running without "
            "`--serve-artifacts`" in resp.text
        )
    finally:
        process.kill()
