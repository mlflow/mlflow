"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""
import json
import os
import sys
import posixpath
import pytest
import shutil
import time
import tempfile
from unittest import mock
import urllib.parse

import mlflow.experiments
from mlflow.exceptions import MlflowException
from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.models import Model

import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
    MLFLOW_USER,
    MLFLOW_RUN_NAME,
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_SOURCE_NAME,
    MLFLOW_PROJECT_ENTRY_POINT,
    MLFLOW_GIT_COMMIT,
)
from mlflow.utils.file_utils import path_to_local_file_uri

from tests.integration.utils import invoke_cli_runner
from tests.tracking.integration_test_utils import _await_server_down_or_die, _init_server

# pylint: disable=unused-argument

# Root directory for all stores (backend or artifact stores) created during this suite
SUITE_ROOT_DIR = tempfile.mkdtemp("test_rest_tracking")
# Root directory for all artifact stores created during this suite
SUITE_ARTIFACT_ROOT_DIR = tempfile.mkdtemp(suffix="artifacts", dir=SUITE_ROOT_DIR)


def _get_sqlite_uri():
    path = path_to_local_file_uri(os.path.join(SUITE_ROOT_DIR, "test-database.bd"))
    path = path[len("file://") :]

    # NB: It looks like windows and posix have different requirements on number of slashes for
    # whatever reason. Windows needs uri like 'sqlite:///C:/path/to/my/file' whereas posix expects
    # sqlite://///path/to/my/file
    prefix = "sqlite://" if sys.platform == "win32" else "sqlite:////"
    return prefix + path


# Backend store URIs to test against
BACKEND_URIS = [
    _get_sqlite_uri(),  # SqlAlchemy
    path_to_local_file_uri(os.path.join(SUITE_ROOT_DIR, "file_store_root")),  # FileStore
]

# Map of backend URI to tuple (server URL, Process). We populate this map by constructing
# a server per backend URI
BACKEND_URI_TO_SERVER_URL_AND_PROC = {
    uri: _init_server(backend_uri=uri, root_artifact_uri=SUITE_ARTIFACT_ROOT_DIR)
    for uri in BACKEND_URIS
}


def pytest_generate_tests(metafunc):
    """
    Automatically parametrize each each fixture/test that depends on `backend_store_uri` with the
    list of backend store URIs.
    """
    if "backend_store_uri" in metafunc.fixturenames:
        metafunc.parametrize("backend_store_uri", BACKEND_URIS)


@pytest.fixture(scope="module", autouse=True)
def server_urls():
    """
    Clean up all servers created for testing in `pytest_generate_tests`
    """
    yield
    for server_url, process in BACKEND_URI_TO_SERVER_URL_AND_PROC.values():
        print("Terminating server at %s..." % (server_url))
        print("type = ", type(process))
        process.terminate()
        _await_server_down_or_die(process)
    shutil.rmtree(SUITE_ROOT_DIR)


@pytest.fixture()
def tracking_server_uri(backend_store_uri):
    url, _ = BACKEND_URI_TO_SERVER_URL_AND_PROC[backend_store_uri]
    return url


@pytest.fixture()
def mlflow_client(tracking_server_uri):
    """Provides an MLflow Tracking API client pointed at the local tracking server."""
    mlflow.set_tracking_uri(tracking_server_uri)
    yield mock.Mock(wraps=MlflowClient(tracking_server_uri))
    mlflow.set_tracking_uri(None)


@pytest.fixture()
def cli_env(tracking_server_uri):
    """Provides an environment for the MLflow CLI pointed at the local tracking server."""
    cli_env = {
        "LC_ALL": "en_US.UTF-8",
        "LANG": "en_US.UTF-8",
        "MLFLOW_TRACKING_URI": tracking_server_uri,
    }
    return cli_env


def test_create_get_list_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment(
        "My Experiment", artifact_location="my_location", tags={"key1": "val1", "key2": "val2"}
    )
    exp = mlflow_client.get_experiment(experiment_id)
    assert exp.name == "My Experiment"
    assert exp.artifact_location == "my_location"
    assert len(exp.tags) == 2
    assert exp.tags["key1"] == "val1"
    assert exp.tags["key2"] == "val2"

    experiments = mlflow_client.list_experiments()
    assert set([e.name for e in experiments]) == {"My Experiment", "Default"}
    mlflow_client.delete_experiment(experiment_id)
    assert set([e.name for e in mlflow_client.list_experiments()]) == {"Default"}
    assert set([e.name for e in mlflow_client.list_experiments(ViewType.ACTIVE_ONLY)]) == {
        "Default"
    }
    assert set([e.name for e in mlflow_client.list_experiments(ViewType.DELETED_ONLY)]) == {
        "My Experiment"
    }
    assert set([e.name for e in mlflow_client.list_experiments(ViewType.ALL)]) == {
        "My Experiment",
        "Default",
    }
    active_exps_paginated = mlflow_client.list_experiments(max_results=1)
    assert set([e.name for e in active_exps_paginated]) == {"Default"}
    assert active_exps_paginated.token is None

    all_exps_paginated = mlflow_client.list_experiments(max_results=1, view_type=ViewType.ALL)
    first_page_names = set([e.name for e in all_exps_paginated])
    all_exps_second_page = mlflow_client.list_experiments(
        max_results=1, view_type=ViewType.ALL, page_token=all_exps_paginated.token
    )
    second_page_names = set([e.name for e in all_exps_second_page])
    assert len(first_page_names) == 1
    assert len(second_page_names) == 1
    assert first_page_names.union(second_page_names) == {"Default", "My Experiment"}


def test_delete_restore_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Deleterious")
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == "active"
    mlflow_client.delete_experiment(experiment_id)
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == "deleted"
    mlflow_client.restore_experiment(experiment_id)
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == "active"


def test_delete_restore_experiment_cli(mlflow_client, cli_env):
    experiment_name = "DeleteriousCLI"
    invoke_cli_runner(
        mlflow.experiments.commands, ["create", "--experiment-name", experiment_name], env=cli_env
    )
    experiment_id = mlflow_client.get_experiment_by_name(experiment_name).experiment_id
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == "active"
    invoke_cli_runner(
        mlflow.experiments.commands, ["delete", "-x", str(experiment_id)], env=cli_env
    )
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == "deleted"
    invoke_cli_runner(
        mlflow.experiments.commands, ["restore", "-x", str(experiment_id)], env=cli_env
    )
    assert mlflow_client.get_experiment(experiment_id).lifecycle_stage == "active"


def test_rename_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment("BadName")
    assert mlflow_client.get_experiment(experiment_id).name == "BadName"
    mlflow_client.rename_experiment(experiment_id, "GoodName")
    assert mlflow_client.get_experiment(experiment_id).name == "GoodName"


def test_rename_experiment_cli(mlflow_client, cli_env):
    bad_experiment_name = "CLIBadName"
    good_experiment_name = "CLIGoodName"

    invoke_cli_runner(
        mlflow.experiments.commands, ["create", "-n", bad_experiment_name], env=cli_env
    )
    experiment_id = mlflow_client.get_experiment_by_name(bad_experiment_name).experiment_id
    assert mlflow_client.get_experiment(experiment_id).name == bad_experiment_name
    invoke_cli_runner(
        mlflow.experiments.commands,
        ["rename", "--experiment-id", str(experiment_id), "--new-name", good_experiment_name],
        env=cli_env,
    )
    assert mlflow_client.get_experiment(experiment_id).name == good_experiment_name


@pytest.mark.parametrize("parent_run_id_kwarg", [None, "my-parent-id"])
def test_create_run_all_args(mlflow_client, parent_run_id_kwarg):
    user = "username"
    source_name = "Hello"
    entry_point = "entry"
    source_version = "abc"
    create_run_kwargs = {
        "start_time": 456,
        "tags": {
            MLFLOW_USER: user,
            MLFLOW_SOURCE_TYPE: "LOCAL",
            MLFLOW_SOURCE_NAME: source_name,
            MLFLOW_PROJECT_ENTRY_POINT: entry_point,
            MLFLOW_GIT_COMMIT: source_version,
            MLFLOW_PARENT_RUN_ID: "7",
            MLFLOW_RUN_NAME: "my name",
            "my": "tag",
            "other": "tag",
        },
    }
    experiment_id = mlflow_client.create_experiment(
        "Run A Lot (parent_run_id=%s)" % (parent_run_id_kwarg)
    )
    created_run = mlflow_client.create_run(experiment_id, **create_run_kwargs)
    run_id = created_run.info.run_id
    print("Run id=%s" % run_id)
    fetched_run = mlflow_client.get_run(run_id)
    for run in [created_run, fetched_run]:
        assert run.info.run_id == run_id
        assert run.info.run_uuid == run_id
        assert run.info.experiment_id == experiment_id
        assert run.info.user_id == user
        assert run.info.start_time == create_run_kwargs["start_time"]
        for tag in create_run_kwargs["tags"]:
            assert tag in run.data.tags
        assert run.data.tags.get(MLFLOW_USER) == user
        assert run.data.tags.get(MLFLOW_RUN_NAME) == "my name"
        assert run.data.tags.get(MLFLOW_PARENT_RUN_ID) == parent_run_id_kwarg or "7"
        assert mlflow_client.list_run_infos(experiment_id) == [run.info]


def test_create_run_defaults(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Run A Little")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    run = mlflow_client.get_run(run_id)
    assert run.info.run_id == run_id
    assert run.info.experiment_id == experiment_id
    assert run.info.user_id == "unknown"


def test_log_metrics_params_tags(mlflow_client, backend_store_uri):
    experiment_id = mlflow_client.create_experiment("Oh My")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    mlflow_client.log_metric(run_id, key="metric", value=123.456, timestamp=789, step=2)
    mlflow_client.log_metric(run_id, key="nan_metric", value=float("nan"))
    mlflow_client.log_metric(run_id, key="inf_metric", value=float("inf"))
    mlflow_client.log_metric(run_id, key="-inf_metric", value=-float("inf"))
    mlflow_client.log_metric(run_id, key="stepless-metric", value=987.654, timestamp=321)
    mlflow_client.log_param(run_id, "param", "value")
    mlflow_client.set_tag(run_id, "taggity", "do-dah")
    run = mlflow_client.get_run(run_id)
    assert run.data.metrics.get("metric") == 123.456
    import math

    assert math.isnan(run.data.metrics.get("nan_metric"))
    assert run.data.metrics.get("inf_metric") >= 1.7976931348623157e308
    assert run.data.metrics.get("-inf_metric") <= -1.7976931348623157e308
    assert run.data.metrics.get("stepless-metric") == 987.654
    assert run.data.params.get("param") == "value"
    assert run.data.tags.get("taggity") == "do-dah"
    metric_history0 = mlflow_client.get_metric_history(run_id, "metric")
    assert len(metric_history0) == 1
    metric0 = metric_history0[0]
    assert metric0.key == "metric"
    assert metric0.value == 123.456
    assert metric0.timestamp == 789
    assert metric0.step == 2
    metric_history1 = mlflow_client.get_metric_history(run_id, "stepless-metric")
    assert len(metric_history1) == 1
    metric1 = metric_history1[0]
    assert metric1.key == "stepless-metric"
    assert metric1.value == 987.654
    assert metric1.timestamp == 321
    assert metric1.step == 0


def test_set_experiment_tag(mlflow_client, backend_store_uri):
    experiment_id = mlflow_client.create_experiment("SetExperimentTagTest")
    mlflow_client.set_experiment_tag(experiment_id, "dataset", "imagenet1K")
    experiment = mlflow_client.get_experiment(experiment_id)
    assert "dataset" in experiment.tags and experiment.tags["dataset"] == "imagenet1K"
    # test that updating a tag works
    mlflow_client.set_experiment_tag(experiment_id, "dataset", "birdbike")
    experiment = mlflow_client.get_experiment(experiment_id)
    assert "dataset" in experiment.tags and experiment.tags["dataset"] == "birdbike"
    # test that setting a tag on 1 experiment does not impact another experiment.
    experiment_id_2 = mlflow_client.create_experiment("SetExperimentTagTest2")
    experiment2 = mlflow_client.get_experiment(experiment_id_2)
    assert len(experiment2.tags) == 0
    # test that setting a tag on different experiments maintain different values across experiments
    mlflow_client.set_experiment_tag(experiment_id_2, "dataset", "birds200")
    experiment = mlflow_client.get_experiment(experiment_id)
    experiment2 = mlflow_client.get_experiment(experiment_id_2)
    assert "dataset" in experiment.tags and experiment.tags["dataset"] == "birdbike"
    assert "dataset" in experiment2.tags and experiment2.tags["dataset"] == "birds200"
    # test can set multi-line tags
    mlflow_client.set_experiment_tag(experiment_id, "multiline tag", "value2\nvalue2\nvalue2")
    experiment = mlflow_client.get_experiment(experiment_id)
    assert (
        "multiline tag" in experiment.tags
        and experiment.tags["multiline tag"] == "value2\nvalue2\nvalue2"
    )


def test_delete_tag(mlflow_client, backend_store_uri):
    experiment_id = mlflow_client.create_experiment("DeleteTagExperiment")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    mlflow_client.log_metric(run_id, key="metric", value=123.456, timestamp=789, step=2)
    mlflow_client.log_metric(run_id, key="stepless-metric", value=987.654, timestamp=321)
    mlflow_client.log_param(run_id, "param", "value")
    mlflow_client.set_tag(run_id, "taggity", "do-dah")
    run = mlflow_client.get_run(run_id)
    assert "taggity" in run.data.tags and run.data.tags["taggity"] == "do-dah"
    mlflow_client.delete_tag(run_id, "taggity")
    run = mlflow_client.get_run(run_id)
    assert "taggity" not in run.data.tags
    with pytest.raises(MlflowException, match=r"Run .+ not found"):
        mlflow_client.delete_tag("fake_run_id", "taggity")
    with pytest.raises(MlflowException, match="No tag with name: fakeTag"):
        mlflow_client.delete_tag(run_id, "fakeTag")
    mlflow_client.delete_run(run_id)
    with pytest.raises(MlflowException, match=f"The run {run_id} must be in"):
        mlflow_client.delete_tag(run_id, "taggity")


def test_log_batch(mlflow_client, backend_store_uri):
    experiment_id = mlflow_client.create_experiment("Batch em up")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    mlflow_client.log_batch(
        run_id=run_id,
        metrics=[Metric("metric", 123.456, 789, 3)],
        params=[Param("param", "value")],
        tags=[RunTag("taggity", "do-dah")],
    )
    run = mlflow_client.get_run(run_id)
    assert run.data.metrics.get("metric") == 123.456
    assert run.data.params.get("param") == "value"
    assert run.data.tags.get("taggity") == "do-dah"
    metric_history = mlflow_client.get_metric_history(run_id, "metric")
    assert len(metric_history) == 1
    metric = metric_history[0]
    assert metric.key == "metric"
    assert metric.value == 123.456
    assert metric.timestamp == 789
    assert metric.step == 3


@pytest.mark.allow_infer_pip_requirements_fallback
def test_log_model(mlflow_client, backend_store_uri):
    experiment_id = mlflow_client.create_experiment("Log models")
    with TempDir(chdr=True):
        mlflow.set_experiment("Log models")
        model_paths = ["model/path/{}".format(i) for i in range(3)]
        with mlflow.start_run(experiment_id=experiment_id) as run:
            for i, m in enumerate(model_paths):
                mlflow.pyfunc.log_model(m, loader_module="mlflow.pyfunc")
                mlflow.pyfunc.save_model(
                    m,
                    mlflow_model=Model(artifact_path=m, run_id=run.info.run_id),
                    loader_module="mlflow.pyfunc",
                )
                model = Model.load(os.path.join(m, "MLmodel"))
                run = mlflow.get_run(run.info.run_id)
                tag = run.data.tags["mlflow.log-model.history"]
                models = json.loads(tag)
                model.utc_time_created = models[i]["utc_time_created"]

                history_model_meta = models[i].copy()
                original_model_uuid = history_model_meta.pop("model_uuid")
                model_meta = model.to_dict().copy()
                new_model_uuid = model_meta.pop(("model_uuid"))
                assert history_model_meta == model_meta
                assert original_model_uuid != new_model_uuid
                assert len(models) == i + 1
                for j in range(0, i + 1):
                    assert models[j]["artifact_path"] == model_paths[j]


def test_set_terminated_defaults(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Terminator 1")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    assert mlflow_client.get_run(run_id).info.status == "RUNNING"
    assert mlflow_client.get_run(run_id).info.end_time is None
    mlflow_client.set_terminated(run_id)
    assert mlflow_client.get_run(run_id).info.status == "FINISHED"
    assert mlflow_client.get_run(run_id).info.end_time <= int(time.time() * 1000)


def test_set_terminated_status(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Terminator 2")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    assert mlflow_client.get_run(run_id).info.status == "RUNNING"
    assert mlflow_client.get_run(run_id).info.end_time is None
    mlflow_client.set_terminated(run_id, "FAILED")
    assert mlflow_client.get_run(run_id).info.status == "FAILED"
    assert mlflow_client.get_run(run_id).info.end_time <= int(time.time() * 1000)


def test_artifacts(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Art In Fact")
    experiment_info = mlflow_client.get_experiment(experiment_id)
    assert experiment_info.artifact_location.startswith(
        path_to_local_file_uri(SUITE_ARTIFACT_ROOT_DIR)
    )
    artifact_path = urllib.parse.urlparse(experiment_info.artifact_location).path
    assert posixpath.split(artifact_path)[-1] == experiment_id

    created_run = mlflow_client.create_run(experiment_id)
    assert created_run.info.artifact_uri.startswith(experiment_info.artifact_location)
    run_id = created_run.info.run_id
    src_dir = tempfile.mkdtemp("test_artifacts_src")
    src_file = os.path.join(src_dir, "my.file")
    with open(src_file, "w") as f:
        f.write("Hello, World!")
    mlflow_client.log_artifact(run_id, src_file, None)
    mlflow_client.log_artifacts(run_id, src_dir, "dir")

    root_artifacts_list = mlflow_client.list_artifacts(run_id)
    assert set([a.path for a in root_artifacts_list]) == {"my.file", "dir"}

    dir_artifacts_list = mlflow_client.list_artifacts(run_id, "dir")
    assert set([a.path for a in dir_artifacts_list]) == {"dir/my.file"}

    all_artifacts = mlflow_client.download_artifacts(run_id, ".")
    assert open("%s/my.file" % all_artifacts, "r").read() == "Hello, World!"
    assert open("%s/dir/my.file" % all_artifacts, "r").read() == "Hello, World!"

    dir_artifacts = mlflow_client.download_artifacts(run_id, "dir")
    assert open("%s/my.file" % dir_artifacts, "r").read() == "Hello, World!"


def test_search_pagination(mlflow_client, backend_store_uri):
    experiment_id = mlflow_client.create_experiment("search_pagination")
    runs = [mlflow_client.create_run(experiment_id, start_time=1).info.run_id for _ in range(0, 10)]
    runs = sorted(runs)
    result = mlflow_client.search_runs([experiment_id], max_results=4, page_token=None)
    assert [r.info.run_id for r in result] == runs[0:4]
    assert result.token is not None
    result = mlflow_client.search_runs([experiment_id], max_results=4, page_token=result.token)
    assert [r.info.run_id for r in result] == runs[4:8]
    assert result.token is not None
    result = mlflow_client.search_runs([experiment_id], max_results=4, page_token=result.token)
    assert [r.info.run_id for r in result] == runs[8:]
    assert result.token is None


def test_get_experiment_by_name(mlflow_client, backend_store_uri):
    name = "test_get_experiment_by_name"
    experiment_id = mlflow_client.create_experiment(name)
    res = mlflow_client.get_experiment_by_name(name)
    assert res.experiment_id == experiment_id
    assert res.name == name
    assert mlflow_client.get_experiment_by_name("idontexist") is None
    mlflow_client.list_experiments.assert_not_called()


def test_get_experiment(mlflow_client, backend_store_uri):
    name = "test_get_experiment"
    experiment_id = mlflow_client.create_experiment(name)
    res = mlflow_client.get_experiment(experiment_id)
    assert res.experiment_id == experiment_id
    assert res.name == name
