import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import numpy as np
import pandas as pd
import pytest
import requests
from botocore.stub import Stubber
from click.testing import CliRunner

import mlflow
from mlflow import pyfunc
from mlflow.cli import cli, doctor, gc, server
from mlflow.data import numpy_dataset
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.server import handlers
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
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


def test_server_uvicorn_options():
    """Test that uvicorn options are properly handled."""
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        # Test default behavior (uvicorn should be used when no server options specified)
        CliRunner().invoke(server)
        run_server_mock.assert_called_once_with(
            file_store_path=mock.ANY,
            registry_store_uri=mock.ANY,
            default_artifact_root=mock.ANY,
            serve_artifacts=mock.ANY,
            artifacts_only=mock.ANY,
            artifacts_destination=mock.ANY,
            host="127.0.0.1",
            port=5000,
            static_prefix=None,
            workers=None,
            gunicorn_opts=None,
            waitress_opts=None,
            expose_prometheus=None,
            app_name=None,
            uvicorn_opts=None,
            env_file=None,
        )

    with mock.patch("mlflow.server._run_server") as run_server_mock:
        # Test with uvicorn-opts - use different options than dev mode
        CliRunner().invoke(server, ["--uvicorn-opts", "--loop asyncio --limit-concurrency 100"])
        run_server_mock.assert_called_once_with(
            file_store_path=mock.ANY,
            registry_store_uri=mock.ANY,
            default_artifact_root=mock.ANY,
            serve_artifacts=mock.ANY,
            artifacts_only=mock.ANY,
            artifacts_destination=mock.ANY,
            host="127.0.0.1",
            port=5000,
            static_prefix=None,
            workers=None,
            gunicorn_opts=None,
            waitress_opts=None,
            expose_prometheus=None,
            app_name=None,
            uvicorn_opts="--loop asyncio --limit-concurrency 100",
            env_file=None,
        )


@pytest.mark.skipif(is_windows(), reason="--dev mode is not supported on Windows")
def test_server_dev_mode():
    """Test that --dev flag sets proper uvicorn options."""
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        # Test with --dev flag (should set uvicorn opts)
        CliRunner().invoke(server, ["--dev"])
        run_server_mock.assert_called_once_with(
            file_store_path=mock.ANY,
            registry_store_uri=mock.ANY,
            default_artifact_root=mock.ANY,
            serve_artifacts=mock.ANY,
            artifacts_only=mock.ANY,
            artifacts_destination=mock.ANY,
            host="127.0.0.1",
            port=5000,
            static_prefix=None,
            workers=None,
            gunicorn_opts=None,
            waitress_opts=None,
            expose_prometheus=None,
            app_name=None,
            uvicorn_opts="--reload --log-level debug",
            env_file=None,
        )


@pytest.mark.skipif(is_windows(), reason="Gunicorn is not supported on Windows")
def test_server_gunicorn_options():
    """Test that gunicorn options are properly handled."""
    with mock.patch("mlflow.server._run_server") as run_server_mock:
        # Test that gunicorn-opts disables uvicorn
        CliRunner().invoke(server, ["--gunicorn-opts", "--timeout 120 --max-requests 1000"])
        run_server_mock.assert_called_once_with(
            file_store_path=mock.ANY,
            registry_store_uri=mock.ANY,
            default_artifact_root=mock.ANY,
            serve_artifacts=mock.ANY,
            artifacts_only=mock.ANY,
            artifacts_destination=mock.ANY,
            host="127.0.0.1",
            port=5000,
            static_prefix=None,
            workers=None,
            gunicorn_opts="--timeout 120 --max-requests 1000",
            waitress_opts=None,
            expose_prometheus=None,
            app_name=None,
            uvicorn_opts=None,
            env_file=None,
        )

    # Test conflicting options
    result = CliRunner().invoke(
        server, ["--uvicorn-opts", "--reload", "--gunicorn-opts", "--log-level debug"]
    )
    assert result.exit_code != 0
    assert "Cannot specify multiple server options" in result.output


def test_server_mlflow_artifacts_options():
    handlers._tracking_store = None
    handlers._model_registry_store = None
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
    with (
        mock.patch("mlflow.server._run_server") as run_server_mock,
        mock.patch("mlflow.store.tracking.sqlalchemy_store.SqlAlchemyStore"),
        mock.patch("mlflow.store.model_registry.sqlalchemy_store.SqlAlchemyStore"),
    ):
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

    with (
        mock.patch("mlflow.server._run_server") as run_server_mock,
        mock.patch("mlflow.store.tracking.file_store.FileStore") as tracking_store,
        mock.patch(
            "mlflow.store.model_registry.sqlalchemy_store.SqlAlchemyStore"
        ) as registry_store,
    ):
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
    store.delete_run(run.info.run_id)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "mlflow",
            "gc",
            "--backend-store-uri",
            sqlite_store[1],
        ]
    )
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 0
    with pytest.raises(MlflowException, match=r"Run .+ not found"):
        store.get_run(run.info.run_id)

    artifact_path = url2pathname(unquote(urlparse(run.info.artifact_uri).path))
    assert not os.path.exists(artifact_path)


def test_mlflow_gc_sqlite_older_than(sqlite_store):
    store = sqlite_store[0]
    run = _create_run_in_store(store)
    store.delete_run(run.info.run_id)
    with pytest.raises(subprocess.CalledProcessError, match=r".+") as exp:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "gc",
                "--backend-store-uri",
                sqlite_store[1],
                "--older-than",
                "10d10h10m10s",
                "--run-ids",
                run.info.run_id,
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
            sys.executable,
            "-m",
            "mlflow",
            "gc",
            "--backend-store-uri",
            sqlite_store[1],
            "--older-than",
            "1s",
            "--run-ids",
            run.info.run_id,
        ]
    )
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 0


@pytest.mark.parametrize("create_artifacts_in_run", [True, False])
def test_mlflow_gc_file_store(file_store, create_artifacts_in_run):
    store = file_store[0]
    run = _create_run_in_store(store, create_artifacts=create_artifacts_in_run)
    store.delete_run(run.info.run_id)
    subprocess.check_output(
        [sys.executable, "-m", "mlflow", "gc", "--backend-store-uri", file_store[1]]
    )
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 0
    with pytest.raises(MlflowException, match=r"Run .+ not found"):
        store.get_run(run.info.run_id)

    artifact_path = url2pathname(unquote(urlparse(run.info.artifact_uri).path))
    assert not os.path.exists(artifact_path)


def test_mlflow_gc_file_store_passing_explicit_run_ids(file_store):
    store = file_store[0]
    run = _create_run_in_store(store)
    store.delete_run(run.info.run_id)
    subprocess.check_output(
        [
            sys.executable,
            "-m",
            "mlflow",
            "gc",
            "--backend-store-uri",
            file_store[1],
            "--run-ids",
            run.info.run_id,
        ]
    )
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 0
    with pytest.raises(MlflowException, match=r"Run .+ not found"):
        store.get_run(run.info.run_id)


def test_mlflow_gc_not_deleted_run(file_store):
    store = file_store[0]
    run = _create_run_in_store(store)
    with pytest.raises(subprocess.CalledProcessError, match=r".+"):
        subprocess.check_output(
            [
                sys.executable,
                "-m",
                "mlflow",
                "gc",
                "--backend-store-uri",
                file_store[1],
                "--run-ids",
                run.info.run_id,
            ]
        )
    runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
    assert len(runs) == 1


def test_mlflow_gc_file_store_older_than(file_store):
    store = file_store[0]
    run = _create_run_in_store(store)
    store.delete_run(run.info.run_id)
    with pytest.raises(subprocess.CalledProcessError, match=r".+") as exp:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "gc",
                "--backend-store-uri",
                file_store[1],
                "--older-than",
                "10d10h10m10s",
                "--run-ids",
                run.info.run_id,
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
            sys.executable,
            "-m",
            "mlflow",
            "gc",
            "--backend-store-uri",
            file_store[1],
            "--older-than",
            "1s",
            "--run-ids",
            run.info.run_id,
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


@pytest.fixture
def sqlite_store_with_s3_artifact_repository():
    fd, temp_dbfile = tempfile.mkstemp()
    # Close handle immediately so that we can remove the file later on in Windows
    os.close(fd)
    db_uri = f"sqlite:///{temp_dbfile}"
    s3_uri = "s3://mlflow"
    store = SqlAlchemyStore(db_uri, s3_uri)

    yield (store, db_uri, s3_uri)

    os.remove(temp_dbfile)


def test_mlflow_gc_sqlite_with_s3_artifact_repository(
    sqlite_store_with_s3_artifact_repository,
):
    store = sqlite_store_with_s3_artifact_repository[0]
    run = _create_run_in_store(store, create_artifacts=False)
    store.delete_run(run.info.run_id)

    artifact_repo = get_artifact_repository(run.info.artifact_uri)
    bucket, dest_path = artifact_repo.parse_s3_compliant_uri(run.info.artifact_uri)
    fake_artifact_path = os.path.join(dest_path, "fake_artifact.txt")
    with Stubber(artifact_repo._get_s3_client()) as s3_stubber:
        s3_stubber.add_response(
            "list_objects_v2",
            {"Contents": [{"Key": fake_artifact_path}]},
            {"Bucket": bucket, "Prefix": dest_path},
        )
        s3_stubber.add_response(
            "delete_objects",
            {"Deleted": [{"Key": fake_artifact_path}]},
            {"Bucket": bucket, "Delete": {"Objects": [{"Key": fake_artifact_path}]}},
        )

        CliRunner().invoke(
            gc,
            [
                "--backend-store-uri",
                sqlite_store_with_s3_artifact_repository[1],
                "--artifacts-destination",
                sqlite_store_with_s3_artifact_repository[2],
            ],
            catch_exceptions=False,
        )

        runs = store.search_runs(experiment_ids=["0"], filter_string="", run_view_type=ViewType.ALL)
        assert len(runs) == 0
        with pytest.raises(MlflowException, match=r"Run .+ not found"):
            store.get_run(run.info.run_id)


@pytest.mark.skip(reason="mlserver is incompatible with the latest version of pydantic")
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
                name="model",
                python_model=model,
                extra_pip_requirements=[
                    "mlserver>=1.2.0,!=1.3.1",
                    "mlserver-mlflow>=1.2.0,!=1.3.1",
                    PROTOBUF_REQUIREMENT,
                ],
            )
        else:
            mlflow.pyfunc.log_model(name="model", python_model=model)
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


def test_mlflow_artifact_list_in_artifacts_only_mode(tmp_path: Path):
    port = get_safe_port()
    cmd = ["mlflow", "server", "--port", str(port), "--artifacts-only"]
    with subprocess.Popen(cmd, cwd=tmp_path) as process:
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
    with (
        mock.patch("mlflow.server._run_server") as run_server_mock,
        mock.patch("mlflow.store.tracking.sqlalchemy_store.SqlAlchemyStore"),
        mock.patch("mlflow.store.model_registry.sqlalchemy_store.SqlAlchemyStore"),
    ):
        result = CliRunner().invoke(
            server,
            ["--artifacts-only", "--backend-store-uri", "sqlite:///my.db"],
            catch_exceptions=False,
        )
        msg = (
            "Usage: server [OPTIONS]\nTry 'server --help' for help.\n\nError: You are starting a "
            "tracking server in `--artifacts-only` mode and have provided a value for "
            "`--backend_store_uri`"
        )
        assert msg in result.output
        assert result.exit_code != 0
        run_server_mock.assert_not_called()


def test_mlflow_ui_is_alias_for_mlflow_server():
    mlflow_ui_stdout = subprocess.check_output(
        [sys.executable, "-m", "mlflow", "ui", "--help"], text=True
    )
    mlflow_server_stdout = subprocess.check_output(
        [sys.executable, "-m", "mlflow", "server", "--help"], text=True
    )
    assert (
        mlflow_ui_stdout.replace("Usage: python -m mlflow ui", "Usage: python -m mlflow server")
        == mlflow_server_stdout
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


def test_env_file_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Setup: Create an experiment using the Python SDK
    # Use file:// URI format for cross-platform compatibility
    mlruns_path = tmp_path / "mlruns"
    test_tracking_uri = mlruns_path.as_uri()  # This creates proper file:// URI
    test_experiment_name = "test_experiment_from_env"

    # Create experiment using SDK
    mlflow.set_tracking_uri(test_tracking_uri)
    mlflow.create_experiment(test_experiment_name)

    # Create a test .env file pointing to this tracking URI
    env_file_path = tmp_path / "test.env"
    env_file_path.write_text(f"MLFLOW_TRACKING_URI={test_tracking_uri}\n")

    runner = CliRunner()

    # Clear the MLflow environment variables to ensure we're testing the loading
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)

    # Ensure variables are not set before running command
    assert "MLFLOW_TRACKING_URI" not in os.environ

    # Use the existing experiments search CLI command with --env-file
    result = runner.invoke(
        cli, ["--env-file", str(env_file_path), "experiments", "search"], catch_exceptions=False
    )

    # Check that the command executed successfully
    assert result.exit_code == 0

    # Verify the experiment we created is found (proves env vars were loaded)
    assert test_experiment_name in result.output

    # Verify the loading message
    assert "Loaded environment variables from:" in result.output
    assert str(env_file_path) in result.output


def test_env_file_loading_invalid_path() -> None:
    runner = CliRunner()

    # Test error handling for non-existent file
    result = runner.invoke(
        cli, ["--env-file", "nonexistent.env", "experiments", "search"], catch_exceptions=False
    )
    assert result.exit_code != 0
    assert "Environment file 'nonexistent.env' does not exist" in result.output


def test_server_with_env_file(tmp_path):
    """Test that --env-file is passed through to uvicorn."""
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_VAR=test_value\n")

    with mock.patch("mlflow.server._run_server") as run_server_mock:
        result = CliRunner().invoke(cli, ["--env-file", str(env_file), "server"])
        assert result.exit_code == 0
        run_server_mock.assert_called_once()
        # Verify env_file parameter is passed
        assert run_server_mock.call_args.kwargs["env_file"] == str(env_file)


def test_mlflow_gc_with_datasets(sqlite_store):
    store = sqlite_store[0]

    mlflow.set_tracking_uri(sqlite_store[1])
    mlflow.set_experiment("dataset")

    dataset = numpy_dataset.from_numpy(np.array([1, 2, 3]))

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id
        mlflow.log_input(dataset)

    experiments = store.search_experiments(view_type=ViewType.ALL)

    # default and datasets
    assert len(experiments) == 2

    store.delete_experiment(experiment_id)

    # the new experiment is only marked as deleted, not removed
    experiments = store.search_experiments(view_type=ViewType.ALL)
    assert len(experiments) == 2

    subprocess.check_call(
        [sys.executable, "-m", "mlflow", "gc", "--backend-store-uri", sqlite_store[1]]
    )
    experiments = store.search_experiments(view_type=ViewType.ALL)

    # only default is left after GC
    assert len(experiments) == 1
    assert experiments[0].experiment_id == "0"
    with pytest.raises(MlflowException, match=f"No Experiment with id={experiment_id} exists"):
        store.get_experiment(experiment_id)
