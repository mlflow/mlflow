import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from unittest import mock
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import numpy as np
import pandas as pd
import pytest
import requests
from click.testing import CliRunner

import mlflow
from mlflow import pyfunc
from mlflow.cli import doctor, gc, server
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.server import handlers
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.os import is_windows
from mlflow.utils.rest_utils import augmented_raise_for_status
from mlflow.utils.time import get_current_time_millis

from tests.helper_functions import PROTOBUF_REQUIREMENT, get_safe_port, pyfunc_serve_and_score_model
from tests.tracking.integration_test_utils import _await_server_up_or_die


@pytest.mark.parametrize("command", ["server"])
def test_mlflow_server_command(command):
    port = get_safe_port()
    cmd = ["mlflow", command, "--port", str(port)]
    process = subprocess.Popen(cmd)
    try:
        _await_server_up_or_die(port)
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
        CliRunner().invoke(server, ["--no-serve-artifacts"])
        run_server_mock.assert_called_once()
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        CliRunner().invoke(server, ["--artifacts-only"])
        run_server_mock.assert_called_once()


@pytest.mark.parametrize("command", [server])
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


@pytest.mark.parametrize("command", [server])
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
    # Clean up the global variables set by the server
    mlflow.set_tracking_uri(None)
    mlflow.set_registry_uri(None)


@pytest.mark.parametrize("command", [server])
def test_registry_store_uri_different_from_tracking_store(command):
    handlers._tracking_store = None
    handlers._model_registry_store = None

    from mlflow.server.handlers import (
        ModelRegistryStoreRegistryWrapper,
        TrackingStoreRegistryWrapper,
    )

    handlers._tracking_store_registry = TrackingStoreRegistryWrapper()
    handlers._model_registry_store_registry = ModelRegistryStoreRegistryWrapper()

    with mock.patch("mlflow.server._run_server") as run_server_mock, mock.patch(
        "mlflow.store.tracking.file_store.FileStore"
    ) as tracking_store, mock.patch(
        "mlflow.store.model_registry.sqlalchemy_store.SqlAlchemyStore"
    ) as registry_store:
        result = CliRunner().invoke(
            command,
            [
                "--backend-store-uri",
                "./mlruns",
                "--registry-store-uri",
                "mysql://user:pwd@host:5432/mydb",
            ],
        )
        assert result.exit_code == 0
        run_server_mock.assert_called()
        tracking_store.assert_called()
        registry_store.assert_called()
    # Clean up the global variables set by the server
    mlflow.set_tracking_uri(None)
    mlflow.set_registry_uri(None)


@pytest.fixture
def sqlite_store():
    fd, temp_dbfile = tempfile.mkstemp()
    # Close handle immediately so that we can remove the file later on in Windows
    os.close(fd)
    db_uri = f"sqlite:///{temp_dbfile}"
    store = SqlAlchemyStore(db_uri, "artifact_folder")
    yield (store, db_uri)
    os.remove(temp_dbfile)
    shutil.rmtree("artifact_folder")


@pytest.fixture
def file_store():
    ROOT_LOCATION = os.path.join(tempfile.gettempdir(), "test_mlflow_gc")
    file_store_uri = f"file:///{ROOT_LOCATION}"
    yield (FileStore(ROOT_LOCATION), file_store_uri)
    shutil.rmtree(ROOT_LOCATION)


def _create_run_in_store(store, create_artifacts=True):
    config = {
        "experiment_id": "0",
        "user_id": "Anderson",
        "start_time": get_current_time_millis(),
        "tags": [],
        "run_name": "name",
    }
    run = store.create_run(**config)
    if create_artifacts:
        artifact_path = url2pathname(unquote(urlparse(run.info.artifact_uri).path))
        if not os.path.exists(artifact_path):
            os.makedirs(artifact_path)
    return run


@pytest.mark.parametrize("create_artifacts_in_run", [True, False])
def test_mlflow_gc_sqlite(sqlite_store, create_artifacts_in_run):
    store = sqlite_store[0]
    run = _create_run_in_store(store, create_artifacts=create_artifacts_in_run)
    store.delete_run(run.info.run_uuid)
    subprocess.check_output(["mlflow", "gc", "--backend-store-uri", sqlite_store[1]])
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 0
    with pytest.raises(MlflowException, match=r"Run .+ not found"):
        store.get_run(run.info.run_uuid)

    artifact_path = url2pathname(unquote(urlparse(run.info.artifact_uri).path))
    assert not os.path.exists(artifact_path)


def test_mlflow_gc_sqlite_older_than(sqlite_store):
    store = sqlite_store[0]
    run = _create_run_in_store(store)
    store.delete_run(run.info.run_uuid)
    with pytest.raises(subprocess.CalledProcessError, match=r".+") as exp:
        subprocess.run(
            [
                "mlflow",
                "gc",
                "--backend-store-uri",
                sqlite_store[1],
                "--older-than",
                "10d10h10m10s",
                "--run-ids",
                run.info.run_uuid,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    assert "is not older than the required age" in exp.value.stderr
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 1

    time.sleep(1)
    subprocess.check_output(
        [
            "mlflow",
            "gc",
            "--backend-store-uri",
            sqlite_store[1],
            "--older-than",
            "1s",
            "--run-ids",
            run.info.run_uuid,
        ]
    )
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 0


@pytest.mark.parametrize("create_artifacts_in_run", [True, False])
def test_mlflow_gc_file_store(file_store, create_artifacts_in_run):
    store = file_store[0]
    run = _create_run_in_store(store, create_artifacts=create_artifacts_in_run)
    store.delete_run(run.info.run_uuid)
    subprocess.check_output(["mlflow", "gc", "--backend-store-uri", file_store[1]])
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 0
    with pytest.raises(MlflowException, match=r"Run .+ not found"):
        store.get_run(run.info.run_uuid)

    artifact_path = url2pathname(unquote(urlparse(run.info.artifact_uri).path))
    assert not os.path.exists(artifact_path)


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


def test_mlflow_gc_file_store_older_than(file_store):
    store = file_store[0]
    run = _create_run_in_store(store)
    store.delete_run(run.info.run_uuid)
    with pytest.raises(subprocess.CalledProcessError, match=r".+") as exp:
        subprocess.run(
            [
                "mlflow",
                "gc",
                "--backend-store-uri",
                file_store[1],
                "--older-than",
                "10d10h10m10s",
                "--run-ids",
                run.info.run_uuid,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    assert "is not older than the required age" in exp.value.stderr
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 1

    time.sleep(1)
    subprocess.check_output(
        [
            "mlflow",
            "gc",
            "--backend-store-uri",
            file_store[1],
            "--older-than",
            "1s",
            "--run-ids",
            run.info.run_uuid,
        ]
    )
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 0


@pytest.mark.parametrize("get_store_details", ["file_store", "sqlite_store"])
def test_mlflow_gc_experiments(get_store_details, request):
    def invoke_gc(*args):
        return CliRunner().invoke(gc, args, catch_exceptions=False)

    store, uri = request.getfixturevalue(get_store_details)
    exp_id_1 = store.create_experiment("1")
    run_id_1 = store.create_run(exp_id_1, user_id="user", start_time=0, tags=[], run_name="1")
    invoke_gc("--backend-store-uri", uri)
    experiments = store.search_experiments(view_type=ViewType.ALL)
    exp_ids = [e.experiment_id for e in experiments]
    runs = store.search_runs(experiment_ids=exp_ids, filter_string="", run_view_type=ViewType.ALL)
    assert sorted(exp_ids) == sorted([exp_id_1, store.DEFAULT_EXPERIMENT_ID])
    assert [r.info.run_id for r in runs] == [run_id_1.info.run_id]

    store.delete_experiment(exp_id_1)
    invoke_gc("--backend-store-uri", uri)
    experiments = store.search_experiments(view_type=ViewType.ALL)
    runs = store.search_runs(experiment_ids=exp_ids, filter_string="", run_view_type=ViewType.ALL)
    assert [e.experiment_id for e in experiments] == [store.DEFAULT_EXPERIMENT_ID]
    assert runs == []

    exp_id_2 = store.create_experiment("2")
    exp_id_3 = store.create_experiment("3")
    store.delete_experiment(exp_id_2)
    store.delete_experiment(exp_id_3)
    invoke_gc("--backend-store-uri", uri, "--experiment-ids", exp_id_2)
    experiments = store.search_experiments(view_type=ViewType.ALL)
    assert sorted([e.experiment_id for e in experiments]) == sorted(
        [exp_id_3, store.DEFAULT_EXPERIMENT_ID]
    )

    with mock.patch("time.time", return_value=0) as mock_time:
        exp_id_4 = store.create_experiment("4")
        store.delete_experiment(exp_id_4)
        mock_time.assert_called()

    invoke_gc("--backend-store-uri", uri, "--older-than", "1d")
    experiments = store.search_experiments(view_type=ViewType.ALL)
    assert sorted([e.experiment_id for e in experiments]) == sorted(
        [exp_id_3, store.DEFAULT_EXPERIMENT_ID]
    )

    invoke_gc("--backend-store-uri", uri, "--experiment-ids", exp_id_3, "--older-than", "0s")
    experiments = store.search_experiments(view_type=ViewType.ALL)
    assert [e.experiment_id for e in experiments] == [store.DEFAULT_EXPERIMENT_ID]

    exp_id_5 = store.create_experiment("5")
    store.delete_experiment(exp_id_5)
    with pytest.raises(MlflowException, match=r"Experiments .+ can be deleted."):
        invoke_gc(
            "--backend-store-uri", uri, "--experiment-ids", exp_id_5, "--older-than", "10d10h10m10s"
        )
    experiments = store.search_experiments(view_type=ViewType.ALL)
    assert sorted([e.experiment_id for e in experiments]) == sorted(
        [exp_id_5, store.DEFAULT_EXPERIMENT_ID]
    )


@pytest.mark.parametrize(
    "enable_mlserver",
    [
        # MLServer is not supported in Windows yet, so let's skip this test in that case.
        # https://github.com/SeldonIO/MLServer/issues/361
        pytest.param(
            True,
            marks=pytest.mark.skipif(is_windows(), reason="MLServer is not supported in Windows"),
        ),
        False,
    ],
)
def test_mlflow_models_serve(enable_mlserver):
    class MyModel(pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return np.array([1, 2, 3])

    model = MyModel()

    with mlflow.start_run():
        if enable_mlserver:
            # We need MLServer to be present on the Conda environment, so we'll
            # add that as an extra requirement.
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model,
                extra_pip_requirements=[
                    "mlserver>=1.2.0,!=1.3.1,<1.4.0",
                    "mlserver-mlflow>=1.2.0,!=1.3.1,<1.4.0",
                    PROTOBUF_REQUIREMENT,
                ],
            )
        else:
            mlflow.pyfunc.log_model(artifact_path="model", python_model=model)
        model_uri = mlflow.get_artifact_uri("model")

    data = pd.DataFrame({"a": [0]})

    extra_args = ["--env-manager", "local"]
    if enable_mlserver:
        extra_args.append("--enable-mlserver")

    scoring_response = pyfunc_serve_and_score_model(
        model_uri=model_uri,
        data=data,
        content_type=pyfunc.scoring_server.CONTENT_TYPE_JSON,
        extra_args=extra_args,
    )
    assert scoring_response.status_code == 200
    served_model_preds = np.array(json.loads(scoring_response.content)["predictions"])
    np.testing.assert_array_equal(served_model_preds, model.predict(data, None))


def test_mlflow_tracking_disabled_in_artifacts_only_mode():
    port = get_safe_port()
    cmd = ["mlflow", "server", "--port", str(port), "--artifacts-only"]
    process = subprocess.Popen(cmd)
    _await_server_up_or_die(port)
    resp = requests.get(f"http://localhost:{port}/api/2.0/mlflow/experiments/search")
    assert (
        "Endpoint: /api/2.0/mlflow/experiments/search disabled due to the mlflow server running "
        "in `--artifacts-only` mode." in resp.text
    )
    process.kill()


def test_mlflow_artifact_list_in_artifacts_only_mode():
    port = get_safe_port()
    cmd = ["mlflow", "server", "--port", str(port), "--artifacts-only"]
    process = subprocess.Popen(cmd)
    try:
        _await_server_up_or_die(port)
        resp = requests.get(f"http://localhost:{port}/api/2.0/mlflow-artifacts/artifacts")
        augmented_raise_for_status(resp)
        assert resp.status_code == 200
        assert resp.text == "{}"
    finally:
        process.kill()


def test_mlflow_artifact_service_unavailable_when_no_server_artifacts_is_specified():
    port = get_safe_port()
    cmd = ["mlflow", "server", "--port", str(port), "--no-serve-artifacts"]
    process = subprocess.Popen(cmd)
    try:
        _await_server_up_or_die(port)
        endpoint = "/api/2.0/mlflow-artifacts/artifacts"
        resp = requests.get(f"http://localhost:{port}{endpoint}")
        assert (
            f"Endpoint: {endpoint} disabled due to the mlflow server running with "
            "`--no-serve-artifacts`" in resp.text
        )
    finally:
        process.kill()


def test_mlflow_artifact_only_prints_warning_for_configs():
    with mock.patch("mlflow.server._run_server") as run_server_mock, mock.patch(
        "mlflow.store.tracking.sqlalchemy_store.SqlAlchemyStore"
    ), mock.patch("mlflow.store.model_registry.sqlalchemy_store.SqlAlchemyStore"):
        result = CliRunner(mix_stderr=False).invoke(
            server,
            ["--artifacts-only", "--backend-store-uri", "sqlite:///my.db"],
            catch_exceptions=False,
        )
        assert result.stderr.startswith(
            "Usage: server [OPTIONS]\nTry 'server --help' for help.\n\nError: You are starting a "
            "tracking server in `--artifacts-only` mode and have provided a value for "
            "`--backend_store_uri`"
        )
        assert result.exit_code != 0
        run_server_mock.assert_not_called()


def test_mlflow_ui_is_alias_for_mlflow_server():
    mlflow_ui_stdout = subprocess.check_output(["mlflow", "ui", "--help"], text=True)
    mlflow_server_stdout = subprocess.check_output(["mlflow", "server", "--help"], text=True)
    assert (
        mlflow_ui_stdout.replace("Usage: mlflow ui", "Usage: mlflow server") == mlflow_server_stdout
    )


def test_cli_with_python_mod():
    stdout = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "mlflow",
            "--version",
        ],
        text=True,
    )
    assert stdout.rstrip().endswith(mlflow.__version__)
    stdout = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--help",
        ],
        text=True,
    )
    assert "mlflow server" in stdout


def test_doctor():
    res = CliRunner().invoke(doctor, catch_exceptions=False)
    assert res.exit_code == 0
