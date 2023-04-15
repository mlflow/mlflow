"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""
import pathlib

import flask
import json
import os
import sys
import posixpath
import logging
import tempfile
import time
import urllib.parse
import requests
from unittest import mock

import pytest

from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
import mlflow.experiments
from mlflow.exceptions import MlflowException
from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.models import Model
import mlflow.pyfunc
from mlflow.server.handlers import validate_path_is_safe
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
    MLFLOW_USER,
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_SOURCE_NAME,
    MLFLOW_PROJECT_ENTRY_POINT,
    MLFLOW_GIT_COMMIT,
)
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.time_utils import get_current_time_millis
from mlflow.utils.os import is_windows

from tests.integration.utils import invoke_cli_runner
from tests.tracking.integration_test_utils import (
    _terminate_server,
    _init_server,
    _send_rest_tracking_post_request,
)


_logger = logging.getLogger(__name__)


@pytest.fixture(params=["file", "sqlalchemy"])
def mlflow_client(request, tmp_path):
    """Provides an MLflow Tracking API client pointed at the local tracking server."""
    if request.param == "file":
        backend_uri = tmp_path.joinpath("file").as_uri()
    elif request.param == "sqlalchemy":
        path = tmp_path.joinpath("sqlalchemy.db").as_uri()
        backend_uri = ("sqlite://" if sys.platform == "win32" else "sqlite:////") + path[
            len("file://") :
        ]

    url, process = _init_server(backend_uri, root_artifact_uri=tmp_path.as_uri())
    yield MlflowClient(url)

    _terminate_server(process)


@pytest.fixture()
def cli_env(mlflow_client):
    """Provides an environment for the MLflow CLI pointed at the local tracking server."""
    cli_env = {
        "LC_ALL": "en_US.UTF-8",
        "LANG": "en_US.UTF-8",
        "MLFLOW_TRACKING_URI": mlflow_client.tracking_uri,
    }
    return cli_env


def create_experiments(client, names):
    return [client.create_experiment(n) for n in names]


def test_create_get_search_experiment(mlflow_client):
    experiment_id = mlflow_client.create_experiment(
        "My Experiment", artifact_location="my_location", tags={"key1": "val1", "key2": "val2"}
    )
    exp = mlflow_client.get_experiment(experiment_id)
    assert exp.name == "My Experiment"
    if is_windows():
        assert exp.artifact_location == pathlib.Path.cwd().joinpath("my_location").as_uri()
    else:
        assert exp.artifact_location == str(pathlib.Path.cwd().joinpath("my_location"))
    assert len(exp.tags) == 2
    assert exp.tags["key1"] == "val1"
    assert exp.tags["key2"] == "val2"

    experiments = mlflow_client.search_experiments()
    assert {e.name for e in experiments} == {"My Experiment", "Default"}
    mlflow_client.delete_experiment(experiment_id)
    assert {e.name for e in mlflow_client.search_experiments()} == {"Default"}
    assert {e.name for e in mlflow_client.search_experiments(view_type=ViewType.ACTIVE_ONLY)} == {
        "Default"
    }
    assert {e.name for e in mlflow_client.search_experiments(view_type=ViewType.DELETED_ONLY)} == {
        "My Experiment"
    }
    assert {e.name for e in mlflow_client.search_experiments(view_type=ViewType.ALL)} == {
        "My Experiment",
        "Default",
    }
    active_exps_paginated = mlflow_client.search_experiments(max_results=1)
    assert {e.name for e in active_exps_paginated} == {"Default"}
    assert active_exps_paginated.token is None

    all_exps_paginated = mlflow_client.search_experiments(max_results=1, view_type=ViewType.ALL)
    first_page_names = {e.name for e in all_exps_paginated}
    all_exps_second_page = mlflow_client.search_experiments(
        max_results=1, view_type=ViewType.ALL, page_token=all_exps_paginated.token
    )
    second_page_names = {e.name for e in all_exps_second_page}
    assert len(first_page_names) == 1
    assert len(second_page_names) == 1
    assert first_page_names.union(second_page_names) == {"Default", "My Experiment"}


def test_create_experiment_validation(mlflow_client):
    def assert_bad_request(payload, expected_error_message):
        response = _send_rest_tracking_post_request(
            mlflow_client.tracking_uri,
            "/api/2.0/mlflow/experiments/create",
            payload,
        )
        assert response.status_code == 400
        assert expected_error_message in response.text

    assert_bad_request(
        {
            "name": 123,
        },
        "Invalid value 123 for parameter 'name'",
    )
    assert_bad_request({}, "Missing value for required parameter 'name'")
    assert_bad_request(
        {
            "name": "experiment name",
            "artifact_location": 9.0,
            "tags": [{"key": "key", "value": "value"}],
        },
        "Invalid value 9.0 for parameter 'artifact_location'",
    )
    assert_bad_request(
        {
            "name": "experiment name",
            "artifact_location": "my_location",
            "tags": "5",
        },
        "Invalid value 5 for parameter 'tags'",
    )


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
        "run_name": "my name",
        "tags": {
            MLFLOW_USER: user,
            MLFLOW_SOURCE_TYPE: "LOCAL",
            MLFLOW_SOURCE_NAME: source_name,
            MLFLOW_PROJECT_ENTRY_POINT: entry_point,
            MLFLOW_GIT_COMMIT: source_version,
            MLFLOW_PARENT_RUN_ID: "7",
            "my": "tag",
            "other": "tag",
        },
    }
    experiment_id = mlflow_client.create_experiment(
        "Run A Lot (parent_run_id=%s)" % (parent_run_id_kwarg)
    )
    created_run = mlflow_client.create_run(experiment_id, **create_run_kwargs)
    run_id = created_run.info.run_id
    _logger.info(f"Run id={run_id}")
    fetched_run = mlflow_client.get_run(run_id)
    for run in [created_run, fetched_run]:
        assert run.info.run_id == run_id
        assert run.info.run_uuid == run_id
        assert run.info.experiment_id == experiment_id
        assert run.info.user_id == user
        assert run.info.start_time == create_run_kwargs["start_time"]
        assert run.info.run_name == "my name"
        for tag in create_run_kwargs["tags"]:
            assert tag in run.data.tags
        assert run.data.tags.get(MLFLOW_USER) == user
        assert run.data.tags.get(MLFLOW_PARENT_RUN_ID) == parent_run_id_kwarg or "7"
        assert [run.info for run in mlflow_client.search_runs([experiment_id])] == [run.info]


def test_create_run_defaults(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Run A Little")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    run = mlflow_client.get_run(run_id)
    assert run.info.run_id == run_id
    assert run.info.experiment_id == experiment_id
    assert run.info.user_id == "unknown"


def test_log_metrics_params_tags(mlflow_client):
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

    metric_history = mlflow_client.get_metric_history(run_id, "a_test_accuracy")
    assert metric_history == []


def test_log_metric_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("metrics validation")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    def assert_bad_request(payload, expected_error_message):
        response = _send_rest_tracking_post_request(
            mlflow_client.tracking_uri,
            "/api/2.0/mlflow/runs/log-metric",
            payload,
        )
        assert response.status_code == 400
        assert expected_error_message in response.text

    assert_bad_request(
        {
            "run_id": 31,
            "key": "metric",
            "value": 41,
            "timestamp": 59,
            "step": 26,
        },
        "Invalid value 31 for parameter 'run_id' supplied",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            "key": 31,
            "value": 41,
            "timestamp": 59,
            "step": 26,
        },
        "Invalid value 31 for parameter 'key' supplied",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            "key": "foo",
            "value": 31,
            "timestamp": 59,
            "step": "foo",
        },
        "Invalid value foo for parameter 'step' supplied",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            "key": "foo",
            "value": 31,
            "timestamp": "foo",
            "step": 41,
        },
        "Invalid value foo for parameter 'timestamp' supplied",
    )
    assert_bad_request(
        {
            "run_id": None,
            "key": "foo",
            "value": 31,
            "timestamp": 59,
            "step": 41,
        },
        "Missing value for required parameter 'run_id'",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            # Missing key
            "value": 31,
            "timestamp": 59,
            "step": 41,
        },
        "Missing value for required parameter 'key'",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            "key": None,
            "value": 31,
            "timestamp": 59,
            "step": 41,
        },
        "Missing value for required parameter 'key'",
    )


def test_log_param_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("params validation")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    def assert_bad_request(payload, expected_error_message):
        response = _send_rest_tracking_post_request(
            mlflow_client.tracking_uri,
            "/api/2.0/mlflow/runs/log-parameter",
            payload,
        )
        assert response.status_code == 400
        assert expected_error_message in response.text

    assert_bad_request(
        {
            "run_id": 31,
            "key": "param",
            "value": 41,
        },
        "Invalid value 31 for parameter 'run_id' supplied",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            "key": 31,
            "value": 41,
        },
        "Invalid value 31 for parameter 'key' supplied",
    )


def test_log_param_with_empty_string_as_value(mlflow_client):
    experiment_id = mlflow_client.create_experiment(
        test_log_param_with_empty_string_as_value.__name__
    )
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    mlflow_client.log_param(run_id, "param_key", "")
    assert {"param_key": ""}.items() <= mlflow_client.get_run(run_id).data.params.items()


def test_set_tag_with_empty_string_as_value(mlflow_client):
    experiment_id = mlflow_client.create_experiment(
        test_set_tag_with_empty_string_as_value.__name__
    )
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    mlflow_client.set_tag(run_id, "tag_key", "")
    assert {"tag_key": ""}.items() <= mlflow_client.get_run(run_id).data.tags.items()


def test_log_batch_containing_params_and_tags_with_empty_string_values(mlflow_client):
    experiment_id = mlflow_client.create_experiment(
        test_log_batch_containing_params_and_tags_with_empty_string_values.__name__
    )
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    mlflow_client.log_batch(
        run_id=run_id,
        params=[Param("param_key", "")],
        tags=[RunTag("tag_key", "")],
    )
    assert {"param_key": ""}.items() <= mlflow_client.get_run(run_id).data.params.items()
    assert {"tag_key": ""}.items() <= mlflow_client.get_run(run_id).data.tags.items()


def test_set_tag_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("tags validation")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    def assert_bad_request(payload, expected_error_message):
        response = _send_rest_tracking_post_request(
            mlflow_client.tracking_uri,
            "/api/2.0/mlflow/runs/set-tag",
            payload,
        )
        assert response.status_code == 400
        assert expected_error_message in response.text

    assert_bad_request(
        {
            "run_id": 31,
            "key": "tag",
            "value": 41,
        },
        "Invalid value 31 for parameter 'run_id' supplied",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            "key": "param",
            "value": 41,
        },
        "Invalid value 41 for parameter 'value' supplied",
    )
    assert_bad_request(
        {
            "run_id": run_id,
            # Missing key
            "value": "value",
        },
        "Missing value for required parameter 'key'",
    )

    response = _send_rest_tracking_post_request(
        mlflow_client.tracking_uri,
        "/api/2.0/mlflow/runs/set-tag",
        {
            "run_uuid": run_id,
            "key": "key",
            "value": "value",
        },
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "path",
        "path/",
        "path/to/file",
    ],
)
def test_validate_path_is_safe_good(path):
    validate_path_is_safe(path)


@pytest.mark.parametrize(
    "path",
    [
        "/path",
        "../path",
        "../../path",
        "./../path",
        "path/../to/file",
        "path/../../to/file",
    ],
)
def test_validate_path_is_safe_bad(path):
    with pytest.raises(MlflowException, match="Invalid path"):
        validate_path_is_safe(path)


def test_path_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("tags validation")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    invalid_path = "../path"

    def assert_response(resp):
        assert resp.status_code == 400
        assert response.json() == {
            "error_code": "INVALID_PARAMETER_VALUE",
            "message": f"Invalid path: {invalid_path}",
        }

    response = requests.get(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/artifacts/list",
        params={"run_id": run_id, "path": invalid_path},
    )
    assert_response(response)

    response = requests.get(
        f"{mlflow_client.tracking_uri}/get-artifact",
        params={"run_id": run_id, "path": invalid_path},
    )
    assert_response(response)

    response = requests.get(
        f"{mlflow_client.tracking_uri}//model-versions/get-artifact",
        params={"name": "model", "version": 1, "path": invalid_path},
    )
    assert_response(response)


def test_set_experiment_tag(mlflow_client):
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


def test_set_experiment_tag_with_empty_string_as_value(mlflow_client):
    experiment_id = mlflow_client.create_experiment(
        test_set_experiment_tag_with_empty_string_as_value.__name__
    )
    mlflow_client.set_experiment_tag(experiment_id, "tag_key", "")
    assert {"tag_key": ""}.items() <= mlflow_client.get_experiment(experiment_id).tags.items()


def test_delete_tag(mlflow_client):
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


def test_log_batch(mlflow_client):
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


def test_log_batch_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("log_batch validation")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id

    def assert_bad_request(payload, expected_error_message):
        response = _send_rest_tracking_post_request(
            mlflow_client.tracking_uri,
            "/api/2.0/mlflow/runs/log-batch",
            payload,
        )
        assert response.status_code == 400
        assert expected_error_message in response.text

    for request_parameter in ["metrics", "params", "tags"]:
        assert_bad_request(
            {
                "run_id": run_id,
                request_parameter: "foo",
            },
            f"Invalid value foo for parameter '{request_parameter}' supplied",
        )


@pytest.mark.allow_infer_pip_requirements_fallback
def test_log_model(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Log models")
    with TempDir(chdr=True):
        model_paths = [f"model/path/{i}" for i in range(3)]
        mlflow.set_tracking_uri(mlflow_client.tracking_uri)
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
                new_model_uuid = model_meta.pop("model_uuid")
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
    assert mlflow_client.get_run(run_id).info.end_time <= get_current_time_millis()


def test_set_terminated_status(mlflow_client):
    experiment_id = mlflow_client.create_experiment("Terminator 2")
    created_run = mlflow_client.create_run(experiment_id)
    run_id = created_run.info.run_id
    assert mlflow_client.get_run(run_id).info.status == "RUNNING"
    assert mlflow_client.get_run(run_id).info.end_time is None
    mlflow_client.set_terminated(run_id, "FAILED")
    assert mlflow_client.get_run(run_id).info.status == "FAILED"
    assert mlflow_client.get_run(run_id).info.end_time <= get_current_time_millis()


def test_artifacts(mlflow_client, tmp_path):
    experiment_id = mlflow_client.create_experiment("Art In Fact")
    experiment_info = mlflow_client.get_experiment(experiment_id)
    assert experiment_info.artifact_location.startswith(path_to_local_file_uri(str(tmp_path)))
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
    assert {a.path for a in root_artifacts_list} == {"my.file", "dir"}

    dir_artifacts_list = mlflow_client.list_artifacts(run_id, "dir")
    assert {a.path for a in dir_artifacts_list} == {"dir/my.file"}

    all_artifacts = download_artifacts(
        run_id=run_id, artifact_path=".", tracking_uri=mlflow_client.tracking_uri
    )
    assert open("%s/my.file" % all_artifacts).read() == "Hello, World!"
    assert open("%s/dir/my.file" % all_artifacts).read() == "Hello, World!"

    dir_artifacts = download_artifacts(
        run_id=run_id, artifact_path="dir", tracking_uri=mlflow_client.tracking_uri
    )
    assert open("%s/my.file" % dir_artifacts).read() == "Hello, World!"


def test_search_pagination(mlflow_client):
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


def test_search_validation(mlflow_client):
    experiment_id = mlflow_client.create_experiment("search_validation")
    with pytest.raises(
        MlflowException, match=r"Invalid value 123456789 for parameter 'max_results' supplied"
    ):
        mlflow_client.search_runs([experiment_id], max_results=123456789)


def test_get_experiment_by_name(mlflow_client):
    name = "test_get_experiment_by_name"
    experiment_id = mlflow_client.create_experiment(name)
    res = mlflow_client.get_experiment_by_name(name)
    assert res.experiment_id == experiment_id
    assert res.name == name
    assert mlflow_client.get_experiment_by_name("idontexist") is None


def test_get_experiment(mlflow_client):
    name = "test_get_experiment"
    experiment_id = mlflow_client.create_experiment(name)
    res = mlflow_client.get_experiment(experiment_id)
    assert res.experiment_id == experiment_id
    assert res.name == name


def test_search_experiments(mlflow_client):
    # To ensure the default experiment and non-default experiments have different creation_time
    # for deterministic search results, send a request to the server and initialize the tracking
    # store.
    assert mlflow_client.search_experiments()[0].name == "Default"

    experiments = [
        ("a", {"key": "value"}),
        ("ab", {"key": "vaLue"}),
        ("Abc", None),
    ]
    experiment_ids = []
    for name, tags in experiments:
        # sleep for windows file system current_time precision in Python to enforce
        # deterministic ordering based on last_update_time (creation_time due to no
        # mutation of experiment state)
        time.sleep(0.001)
        experiment_ids.append(mlflow_client.create_experiment(name, tags=tags))

    # filter_string
    experiments = mlflow_client.search_experiments(filter_string="attribute.name = 'a'")
    assert [e.name for e in experiments] == ["a"]
    experiments = mlflow_client.search_experiments(filter_string="attribute.name != 'a'")
    assert [e.name for e in experiments] == ["Abc", "ab", "Default"]
    experiments = mlflow_client.search_experiments(filter_string="name LIKE 'a%'")
    assert [e.name for e in experiments] == ["ab", "a"]
    experiments = mlflow_client.search_experiments(filter_string="tag.key = 'value'")
    assert [e.name for e in experiments] == ["a"]
    experiments = mlflow_client.search_experiments(filter_string="tag.key != 'value'")
    assert [e.name for e in experiments] == ["ab"]
    experiments = mlflow_client.search_experiments(filter_string="tag.key ILIKE '%alu%'")
    assert [e.name for e in experiments] == ["ab", "a"]

    # order_by
    experiments = mlflow_client.search_experiments(order_by=["name DESC"])
    assert [e.name for e in experiments] == ["ab", "a", "Default", "Abc"]

    # max_results
    experiments = mlflow_client.search_experiments(max_results=2)
    assert [e.name for e in experiments] == ["Abc", "ab"]
    # page_token
    experiments = mlflow_client.search_experiments(page_token=experiments.token)
    assert [e.name for e in experiments] == ["a", "Default"]

    # view_type
    time.sleep(0.001)
    mlflow_client.delete_experiment(experiment_ids[1])
    experiments = mlflow_client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    assert [e.name for e in experiments] == ["Abc", "a", "Default"]
    experiments = mlflow_client.search_experiments(view_type=ViewType.DELETED_ONLY)
    assert [e.name for e in experiments] == ["ab"]
    experiments = mlflow_client.search_experiments(view_type=ViewType.ALL)
    assert [e.name for e in experiments] == ["Abc", "ab", "a", "Default"]


def test_get_metric_history_bulk_rejects_invalid_requests(mlflow_client):
    def assert_response(resp, message_part):
        assert resp.status_code == 400
        response_json = resp.json()
        assert response_json.get("error_code") == "INVALID_PARAMETER_VALUE"
        assert message_part in response_json.get("message", "")

    response_no_run_ids_field = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"metric_key": "key"},
    )
    assert_response(
        response_no_run_ids_field,
        "GetMetricHistoryBulk request must specify at least one run_id",
    )

    response_empty_run_ids = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [], "metric_key": "key"},
    )
    assert_response(
        response_empty_run_ids,
        "GetMetricHistoryBulk request must specify at least one run_id",
    )

    response_too_many_run_ids = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [f"id_{i}" for i in range(1000)], "metric_key": "key"},
    )
    assert_response(
        response_too_many_run_ids,
        "GetMetricHistoryBulk request cannot specify more than",
    )

    response_no_metric_key_field = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": ["123"]},
    )
    assert_response(
        response_no_metric_key_field,
        "GetMetricHistoryBulk request must specify a metric_key",
    )


def test_get_metric_history_bulk_returns_expected_metrics_in_expected_order(mlflow_client):
    experiment_id = mlflow_client.create_experiment("get metric history bulk")
    created_run1 = mlflow_client.create_run(experiment_id)
    run_id1 = created_run1.info.run_id
    created_run2 = mlflow_client.create_run(experiment_id)
    run_id2 = created_run2.info.run_id
    created_run3 = mlflow_client.create_run(experiment_id)
    run_id3 = created_run3.info.run_id

    metricA_history = [
        {"key": "metricA", "timestamp": 1, "step": 2, "value": 10.0},
        {"key": "metricA", "timestamp": 1, "step": 3, "value": 11.0},
        {"key": "metricA", "timestamp": 1, "step": 3, "value": 12.0},
        {"key": "metricA", "timestamp": 2, "step": 3, "value": 12.0},
    ]
    for metric in metricA_history:
        mlflow_client.log_metric(run_id1, **metric)
        metric_for_run2 = dict(metric)
        metric_for_run2["value"] += 1.0
        mlflow_client.log_metric(run_id2, **metric_for_run2)

    metricB_history = [
        {"key": "metricB", "timestamp": 7, "step": -2, "value": -100.0},
        {"key": "metricB", "timestamp": 8, "step": 0, "value": 0.0},
        {"key": "metricB", "timestamp": 8, "step": 0, "value": 1.0},
        {"key": "metricB", "timestamp": 9, "step": 1, "value": 12.0},
    ]
    for metric in metricB_history:
        mlflow_client.log_metric(run_id1, **metric)
        metric_for_run2 = dict(metric)
        metric_for_run2["value"] += 1.0
        mlflow_client.log_metric(run_id2, **metric_for_run2)

    response_run1_metricA = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [run_id1], "metric_key": "metricA"},
    )
    assert response_run1_metricA.status_code == 200
    assert response_run1_metricA.json().get("metrics") == [
        {**metric, "run_id": run_id1} for metric in metricA_history
    ]

    response_run2_metricB = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [run_id2], "metric_key": "metricB"},
    )
    assert response_run2_metricB.status_code == 200
    assert response_run2_metricB.json().get("metrics") == [
        {**metric, "run_id": run_id2, "value": metric["value"] + 1.0} for metric in metricB_history
    ]

    response_run1_run2_metricA = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [run_id1, run_id2], "metric_key": "metricA"},
    )
    assert response_run1_run2_metricA.status_code == 200
    assert response_run1_run2_metricA.json().get("metrics") == sorted(
        [{**metric, "run_id": run_id1} for metric in metricA_history]
        + [
            {**metric, "run_id": run_id2, "value": metric["value"] + 1.0}
            for metric in metricA_history
        ],
        key=lambda metric: metric["run_id"],
    )

    response_run1_run2_run_3_metricB = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [run_id1, run_id2, run_id3], "metric_key": "metricB"},
    )
    assert response_run1_run2_run_3_metricB.status_code == 200
    assert response_run1_run2_run_3_metricB.json().get("metrics") == sorted(
        [{**metric, "run_id": run_id1} for metric in metricB_history]
        + [
            {**metric, "run_id": run_id2, "value": metric["value"] + 1.0}
            for metric in metricB_history
        ],
        key=lambda metric: metric["run_id"],
    )


def test_get_metric_history_bulk_respects_max_results(mlflow_client):
    experiment_id = mlflow_client.create_experiment("get metric history bulk")
    run_id = mlflow_client.create_run(experiment_id).info.run_id
    max_results = 2

    metricA_history = [
        {"key": "metricA", "timestamp": 1, "step": 2, "value": 10.0},
        {"key": "metricA", "timestamp": 1, "step": 3, "value": 11.0},
        {"key": "metricA", "timestamp": 1, "step": 3, "value": 12.0},
        {"key": "metricA", "timestamp": 2, "step": 3, "value": 12.0},
    ]
    for metric in metricA_history:
        mlflow_client.log_metric(run_id, **metric)

    response_limited = requests.get(
        f"{mlflow_client.tracking_uri}/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        params={"run_id": [run_id], "metric_key": "metricA", "max_results": max_results},
    )
    assert response_limited.status_code == 200
    assert response_limited.json().get("metrics") == [
        {**metric, "run_id": run_id} for metric in metricA_history[:max_results]
    ]


def test_get_metric_history_bulk_calls_optimized_impl_when_expected(monkeypatch, tmp_path):
    from mlflow.server.handlers import get_metric_history_bulk_handler

    path = path_to_local_file_uri(str(tmp_path.joinpath("sqlalchemy.db")))
    uri = ("sqlite://" if sys.platform == "win32" else "sqlite:////") + path[len("file://") :]
    mock_store = mock.Mock(wraps=SqlAlchemyStore(uri, str(tmp_path)))

    flask_app = flask.Flask("test_flask_app")

    class MockRequestArgs:
        def __init__(self, args_dict):
            self.args_dict = args_dict

        def to_dict(self, flat):
            return self.args_dict

        def get(self, key, default=None):
            return self.args_dict.get(key, default)

    with mock.patch(
        "mlflow.server.handlers._get_tracking_store", return_value=mock_store
    ), flask_app.test_request_context() as mock_context:
        run_ids = [str(i) for i in range(10)]
        mock_context.request.args = MockRequestArgs(
            {
                "run_id": run_ids,
                "metric_key": "mock_key",
            }
        )

        get_metric_history_bulk_handler()

        mock_store.get_metric_history_bulk.assert_called_once_with(
            run_ids=run_ids,
            metric_key="mock_key",
            max_results=25000,
        )


def test_create_model_version_with_path_source(mlflow_client):
    name = "mode"
    mlflow_client.create_registered_model(name)
    exp_id = mlflow_client.create_experiment("test")
    run = mlflow_client.create_run(experiment_id=exp_id)

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri[len("file://") :],
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # run_id is not specified
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri[len("file://") :],
        },
    )
    assert response.status_code == 400
    assert "To use a local path as a model version" in response.json()["message"]

    # run_id is specified but source is not in the run's artifact directory
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "/tmp",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "To use a local path as a model version" in response.json()["message"]


def test_create_model_version_with_file_uri(mlflow_client):
    name = "test"
    mlflow_client.create_registered_model(name)
    exp_id = mlflow_client.create_experiment("test")
    run = mlflow_client.create_run(experiment_id=exp_id)
    assert run.info.artifact_uri.startswith("file://")
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri,
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": f"{run.info.artifact_uri}/model",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": f"{run.info.artifact_uri}/.",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": f"{run.info.artifact_uri}/model/..",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 200

    # run_id is not specified
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": run.info.artifact_uri,
        },
    )
    assert response.status_code == 400
    assert "To use a local path as a model version" in response.json()["message"]

    # run_id is specified but source is not in the run's artifact directory
    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "file:///tmp",
        },
    )
    assert response.status_code == 400
    assert "To use a local path as a model version" in response.json()["message"]

    response = requests.post(
        f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={
            "name": name,
            "source": "file://123.456.789.123/path/to/source",
            "run_id": run.info.run_id,
        },
    )
    assert response.status_code == 400
    assert "MLflow tracking server doesn't allow" in response.json()["message"]


def test_create_model_version_with_file_uri_env_var(tmp_path):
    backend_uri = tmp_path.joinpath("file").as_uri()
    url, process = _init_server(
        backend_uri,
        root_artifact_uri=tmp_path.as_uri(),
        extra_env={"MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE": "true"},
    )
    try:
        mlflow_client = MlflowClient(url)

        name = "test"
        mlflow_client.create_registered_model(name)
        exp_id = mlflow_client.create_experiment("test")
        run = mlflow_client.create_run(experiment_id=exp_id)
        response = requests.post(
            f"{mlflow_client.tracking_uri}/api/2.0/mlflow/model-versions/create",
            json={
                "name": name,
                "source": "file://123.456.789.123/path/to/source",
                "run_id": run.info.run_id,
            },
        )
        assert response.status_code == 200
    finally:
        _terminate_server(process)


def test_logging_model_with_local_artifact_uri(mlflow_client):
    from sklearn.linear_model import LogisticRegression

    mlflow.set_tracking_uri(mlflow_client.tracking_uri)
    with mlflow.start_run() as run:
        assert run.info.artifact_uri.startswith("file://")
        mlflow.sklearn.log_model(LogisticRegression(), "model", registered_model_name="rmn")
        mlflow.pyfunc.load_model("models:/rmn/1")


def test_update_run_name_without_changing_status(mlflow_client):
    experiment_id = mlflow_client.create_experiment("update run name")
    created_run = mlflow_client.create_run(experiment_id)
    mlflow_client.set_terminated(created_run.info.run_id, "FINISHED")

    mlflow_client.update_run(created_run.info.run_id, name="name_abc")
    updated_run_info = mlflow_client.get_run(created_run.info.run_id).info
    assert updated_run_info.run_name == "name_abc"
    assert updated_run_info.status == "FINISHED"
