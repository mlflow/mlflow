"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures we can use the tracking API to communicate with it.
"""
import json
import os
import sys
import posixpath
import pytest
import logging
import tempfile
import urllib.parse
from unittest import mock

import mlflow.experiments
from mlflow.exceptions import MlflowException
from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.models import Model

import mlflow.pyfunc
from mlflow import MlflowClient
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

from tests.integration.utils import invoke_cli_runner
from tests.tracking.integration_test_utils import (
    _await_server_down_or_die,
    _init_server,
    _send_rest_tracking_post_request,
)


_logger = logging.getLogger(__name__)


@pytest.fixture(params=["file", "sqlalchemy"])
def mlflow_client(request, tmp_path):
    """Provides an MLflow Tracking API client pointed at the local tracking server."""
    if request.param == "file":
        uri = path_to_local_file_uri(str(tmp_path.joinpath("file")))
    elif request.param == "sqlalchemy":
        path = path_to_local_file_uri(str(tmp_path.joinpath("sqlalchemy.db")))
        uri = ("sqlite://" if sys.platform == "win32" else "sqlite:////") + path[len("file://") :]

    url, process = _init_server(backend_uri=uri, root_artifact_uri=str(tmp_path))

    yield MlflowClient(url)

    _logger.info(f"Terminating server at {url}...")
    process.terminate()
    _await_server_down_or_die(process)


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
    assert {e.name for e in experiments} == {"My Experiment", "Default"}
    mlflow_client.delete_experiment(experiment_id)
    assert {e.name for e in mlflow_client.list_experiments()} == {"Default"}
    assert {e.name for e in mlflow_client.list_experiments(ViewType.ACTIVE_ONLY)} == {"Default"}
    assert {e.name for e in mlflow_client.list_experiments(ViewType.DELETED_ONLY)} == {
        "My Experiment"
    }
    assert {e.name for e in mlflow_client.list_experiments(ViewType.ALL)} == {
        "My Experiment",
        "Default",
    }
    active_exps_paginated = mlflow_client.list_experiments(max_results=1)
    assert {e.name for e in active_exps_paginated} == {"Default"}
    assert active_exps_paginated.token is None

    all_exps_paginated = mlflow_client.list_experiments(max_results=1, view_type=ViewType.ALL)
    first_page_names = {e.name for e in all_exps_paginated}
    all_exps_second_page = mlflow_client.list_experiments(
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
        assert mlflow_client.list_run_infos(experiment_id) == [run.info]


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

    all_artifacts = mlflow_client.download_artifacts(run_id, ".")
    assert open("%s/my.file" % all_artifacts, "r").read() == "Hello, World!"
    assert open("%s/dir/my.file" % all_artifacts, "r").read() == "Hello, World!"

    dir_artifacts = mlflow_client.download_artifacts(run_id, "dir")
    assert open("%s/my.file" % dir_artifacts, "r").read() == "Hello, World!"


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
    with mock.patch.object(
        MlflowClient,
        "list_experiments",
        side_effect=Exception("should not be called"),
    ):
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
    experiments = [
        ("a", {"key": "value"}),
        ("ab", {"key": "vaLue"}),
        ("Abc", None),
    ]
    experiment_ids = [
        mlflow_client.create_experiment(name, tags=tags) for name, tags in experiments
    ]

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
    mlflow_client.delete_experiment(experiment_ids[1])
    experiments = mlflow_client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    assert [e.name for e in experiments] == ["Abc", "a", "Default"]
    experiments = mlflow_client.search_experiments(view_type=ViewType.DELETED_ONLY)
    assert [e.name for e in experiments] == ["ab"]
    experiments = mlflow_client.search_experiments(view_type=ViewType.ALL)
    assert [e.name for e in experiments] == ["ab", "Abc", "a", "Default"]
