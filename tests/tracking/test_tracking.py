import filecmp
import json
import os
import pathlib
import posixpath
import random
import re
from collections import namedtuple
from datetime import datetime, timezone
from unittest import mock

import pytest
import yaml

import mlflow
from mlflow import MlflowClient, tracking
from mlflow.entities import LifecycleStage, Metric, Param, RunStatus, RunTag, ViewType
from mlflow.environment_variables import (
    MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE,
    MLFLOW_RUN_ID,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.tracking.file_store import FileStore
from mlflow.tracking.fluent import start_run
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.mlflow_tags import (
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_RUN_NAME,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_USER,
)
from mlflow.utils.os import is_windows
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
    MAX_METRICS_PER_BATCH,
    MAX_PARAMS_TAGS_PER_BATCH,
)

MockExperiment = namedtuple("MockExperiment", ["experiment_id", "lifecycle_stage"])


def test_create_experiment():
    with pytest.raises(MlflowException, match="Invalid experiment name"):
        mlflow.create_experiment(None)

    with pytest.raises(MlflowException, match="Invalid experiment name"):
        mlflow.create_experiment("")

    exp_id = mlflow.create_experiment(f"Some random experiment name {random.randint(1, 1e6)}")
    assert exp_id is not None


def test_create_experiment_with_duplicate_name():
    name = "popular_name"
    exp_id = mlflow.create_experiment(name)

    with pytest.raises(MlflowException, match=re.escape(f"Experiment(name={name}) already exists")):
        mlflow.create_experiment(name)

    tracking.MlflowClient().delete_experiment(exp_id)
    with pytest.raises(MlflowException, match=re.escape(f"Experiment(name={name}) already exists")):
        mlflow.create_experiment(name)


def test_create_experiments_with_bad_names():
    # None for name
    with pytest.raises(MlflowException, match="Invalid experiment name: 'None'"):
        mlflow.create_experiment(None)

    # empty string name
    with pytest.raises(MlflowException, match="Invalid experiment name: ''"):
        mlflow.create_experiment("")


@pytest.mark.parametrize("name", [123, 0, -1.2, [], ["A"], {1: 2}])
def test_create_experiments_with_bad_name_types(name):
    with pytest.raises(
        MlflowException,
        match=re.escape(f"Invalid experiment name: {name}. Expects a string."),
    ):
        mlflow.create_experiment(name)


@pytest.mark.usefixtures("reset_active_experiment")
def test_set_experiment_by_name():
    name = "random_exp"
    exp_id = mlflow.create_experiment(name)
    exp1 = mlflow.set_experiment(name)
    assert exp1.experiment_id == exp_id
    with start_run() as run:
        assert run.info.experiment_id == exp_id

    another_name = "another_experiment"
    exp2 = mlflow.set_experiment(another_name)
    with start_run() as another_run:
        assert another_run.info.experiment_id == exp2.experiment_id


@pytest.mark.usefixtures("reset_active_experiment")
def test_set_experiment_by_id():
    name = "random_exp"
    exp_id = mlflow.create_experiment(name)
    active_exp = mlflow.set_experiment(experiment_id=exp_id)
    assert active_exp.experiment_id == exp_id
    with start_run() as run:
        assert run.info.experiment_id == exp_id

    nonexistent_id = "-1337"
    with pytest.raises(MlflowException, match="No Experiment with id=-1337 exists") as exc:
        mlflow.set_experiment(experiment_id=nonexistent_id)
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    with start_run() as run:
        assert run.info.experiment_id == exp_id


def test_set_experiment_parameter_validation():
    with pytest.raises(MlflowException, match="Must specify exactly one") as exc:
        mlflow.set_experiment()
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    with pytest.raises(MlflowException, match="Must specify exactly one") as exc:
        mlflow.set_experiment(None)
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    with pytest.raises(MlflowException, match="Must specify exactly one") as exc:
        mlflow.set_experiment(None, None)
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    with pytest.raises(MlflowException, match="Must specify exactly one") as exc:
        mlflow.set_experiment("name", "id")
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_set_experiment_with_deleted_experiment():
    name = "dead_exp"
    mlflow.set_experiment(name)
    with start_run() as run:
        exp_id = run.info.experiment_id

    tracking.MlflowClient().delete_experiment(exp_id)

    with pytest.raises(MlflowException, match="Cannot set a deleted experiment") as exc:
        mlflow.set_experiment(name)
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    with pytest.raises(MlflowException, match="Cannot set a deleted experiment") as exc:
        mlflow.set_experiment(experiment_id=exp_id)
    assert exc.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.usefixtures("reset_active_experiment")
def test_set_experiment_with_zero_id():
    mock_experiment = MockExperiment(experiment_id=0, lifecycle_stage=LifecycleStage.ACTIVE)
    with mock.patch.object(
        MlflowClient,
        "get_experiment_by_name",
        mock.Mock(return_value=mock_experiment),
    ) as get_experiment_by_name_mock, mock.patch.object(
        MlflowClient, "create_experiment"
    ) as create_experiment_mock:
        mlflow.set_experiment("my_exp")
        get_experiment_by_name_mock.assert_called_once()
        create_experiment_mock.assert_not_called()


def test_start_run_context_manager():
    with start_run() as first_run:
        first_uuid = first_run.info.run_id
        # Check that start_run() causes the run information to be persisted in the store
        persisted_run = tracking.MlflowClient().get_run(first_uuid)
        assert persisted_run is not None
        assert persisted_run.info == first_run.info
    finished_run = tracking.MlflowClient().get_run(first_uuid)
    assert finished_run.info.status == RunStatus.to_string(RunStatus.FINISHED)
    # Launch a separate run that fails, verify the run status is FAILED and the run UUID is
    # different
    with pytest.raises(Exception, match="Failing run!"):
        with start_run() as second_run:
            raise Exception("Failing run!")
    second_run_id = second_run.info.run_id
    assert second_run_id != first_uuid
    finished_run2 = tracking.MlflowClient().get_run(second_run_id)
    assert finished_run2.info.status == RunStatus.to_string(RunStatus.FAILED)


def test_start_and_end_run():
    # Use the start_run() and end_run() APIs without a `with` block, verify they work.

    with start_run() as active_run:
        mlflow.log_metric("name_1", 25)
    finished_run = tracking.MlflowClient().get_run(active_run.info.run_id)
    # Validate metrics
    assert len(finished_run.data.metrics) == 1
    assert finished_run.data.metrics["name_1"] == 25


def test_metric_timestamp():
    with mlflow.start_run() as active_run:
        mlflow.log_metric("name_1", 25)
        mlflow.log_metric("name_1", 30)
        run_id = active_run.info.run_uuid
    # Check that metric timestamps are between run start and finish
    client = MlflowClient()
    history = client.get_metric_history(run_id, "name_1")
    finished_run = client.get_run(run_id)
    assert len(history) == 2
    assert all(
        m.timestamp >= finished_run.info.start_time and m.timestamp <= finished_run.info.end_time
        for m in history
    )


def test_log_batch():
    expected_metrics = {"metric-key0": 1.0, "metric-key1": 4.0}
    expected_params = {"param-key0": "param-val0", "param-key1": "param-val1"}
    exact_expected_tags = {"tag-key0": "tag-val0", "tag-key1": "tag-val1"}
    approx_expected_tags = {
        MLFLOW_USER,
        MLFLOW_SOURCE_NAME,
        MLFLOW_SOURCE_TYPE,
        MLFLOW_RUN_NAME,
    }

    t = get_current_time_millis()
    sorted_expected_metrics = sorted(expected_metrics.items(), key=lambda kv: kv[0])
    metrics = [
        Metric(key=key, value=value, timestamp=t, step=i)
        for i, (key, value) in enumerate(sorted_expected_metrics)
    ]
    params = [Param(key=key, value=value) for key, value in expected_params.items()]
    tags = [RunTag(key=key, value=value) for key, value in exact_expected_tags.items()]

    with start_run() as active_run:
        run_id = active_run.info.run_id
        MlflowClient().log_batch(run_id=run_id, metrics=metrics, params=params, tags=tags)
    client = tracking.MlflowClient()
    finished_run = client.get_run(run_id)
    # Validate metrics
    assert len(finished_run.data.metrics) == 2
    for key, value in finished_run.data.metrics.items():
        assert expected_metrics[key] == value
    metric_history0 = client.get_metric_history(run_id, "metric-key0")
    assert {(m.value, m.timestamp, m.step) for m in metric_history0} == {(1.0, t, 0)}
    metric_history1 = client.get_metric_history(run_id, "metric-key1")
    assert {(m.value, m.timestamp, m.step) for m in metric_history1} == {(4.0, t, 1)}

    # Validate tags (for automatically-set tags)
    assert len(finished_run.data.tags) == len(exact_expected_tags) + len(approx_expected_tags)
    for tag_key, tag_value in finished_run.data.tags.items():
        if tag_key in approx_expected_tags:
            pass
        else:
            assert exact_expected_tags[tag_key] == tag_value
    # Validate params
    assert finished_run.data.params == expected_params
    # test that log_batch works with fewer params
    new_tags = {"1": "2", "3": "4", "5": "6"}
    tags = [RunTag(key=key, value=value) for key, value in new_tags.items()]
    client.log_batch(run_id=run_id, tags=tags)
    finished_run_2 = client.get_run(run_id)
    # Validate tags (for automatically-set tags)
    assert len(finished_run_2.data.tags) == len(finished_run.data.tags) + 3
    for tag_key, tag_value in finished_run_2.data.tags.items():
        if tag_key in new_tags:
            assert new_tags[tag_key] == tag_value


def test_log_batch_with_many_elements():
    num_metrics = MAX_METRICS_PER_BATCH * 2
    num_params = num_tags = MAX_PARAMS_TAGS_PER_BATCH * 2
    expected_metrics = {f"metric-key{i}": float(i) for i in range(num_metrics)}
    expected_params = {f"param-key{i}": f"param-val{i}" for i in range(num_params)}
    exact_expected_tags = {f"tag-key{i}": f"tag-val{i}" for i in range(num_tags)}

    t = get_current_time_millis()
    sorted_expected_metrics = sorted(expected_metrics.items(), key=lambda kv: kv[1])
    metrics = [
        Metric(key=key, value=value, timestamp=t, step=i)
        for i, (key, value) in enumerate(sorted_expected_metrics)
    ]
    params = [Param(key=key, value=value) for key, value in expected_params.items()]
    tags = [RunTag(key=key, value=value) for key, value in exact_expected_tags.items()]

    with start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.tracking.MlflowClient().log_batch(
            run_id=run_id, metrics=metrics, params=params, tags=tags
        )
    client = tracking.MlflowClient()
    finished_run = client.get_run(run_id)
    # Validate metrics
    assert expected_metrics == finished_run.data.metrics
    for i in range(num_metrics):
        metric_history = client.get_metric_history(run_id, f"metric-key{i}")
        assert {(m.value, m.timestamp, m.step) for m in metric_history} == {(float(i), t, i)}

    # Validate tags
    logged_tags = finished_run.data.tags
    for tag_key, tag_value in exact_expected_tags.items():
        assert logged_tags[tag_key] == tag_value

    # Validate params
    assert finished_run.data.params == expected_params


def test_log_metric():
    with start_run() as active_run, mock.patch("time.time") as time_mock:
        time_mock.side_effect = [123 for _ in range(100)]
        run_id = active_run.info.run_id
        mlflow.log_metric("name_1", 25)
        mlflow.log_metric("name_2", -3)
        mlflow.log_metric("name_1", 30, 5)
        mlflow.log_metric("name_1", 40, -2)
        mlflow.log_metric("nested/nested/name", 40)
    finished_run = tracking.MlflowClient().get_run(run_id)
    # Validate metrics
    assert len(finished_run.data.metrics) == 3
    expected_pairs = {"name_1": 30, "name_2": -3, "nested/nested/name": 40}
    for key, value in finished_run.data.metrics.items():
        assert expected_pairs[key] == value
    client = tracking.MlflowClient()
    metric_history_name1 = client.get_metric_history(run_id, "name_1")
    assert {(m.value, m.timestamp, m.step) for m in metric_history_name1} == {
        (25, 123 * 1000, 0),
        (30, 123 * 1000, 5),
        (40, 123 * 1000, -2),
    }
    metric_history_name2 = client.get_metric_history(run_id, "name_2")
    assert {(m.value, m.timestamp, m.step) for m in metric_history_name2} == {(-3, 123 * 1000, 0)}


def test_log_metrics_uses_millisecond_timestamp_resolution_fluent():
    with start_run() as active_run, mock.patch("time.time") as time_mock:
        time_mock.side_effect = lambda: 123
        mlflow.log_metrics({"name_1": 25, "name_2": -3})
        mlflow.log_metrics({"name_1": 30})
        mlflow.log_metrics({"name_1": 40})
        run_id = active_run.info.run_id

    client = tracking.MlflowClient()
    metric_history_name1 = client.get_metric_history(run_id, "name_1")
    assert {(m.value, m.timestamp) for m in metric_history_name1} == {
        (25, 123 * 1000),
        (30, 123 * 1000),
        (40, 123 * 1000),
    }
    metric_history_name2 = client.get_metric_history(run_id, "name_2")
    assert {(m.value, m.timestamp) for m in metric_history_name2} == {(-3, 123 * 1000)}


def test_log_metrics_uses_millisecond_timestamp_resolution_client():
    with start_run() as active_run, mock.patch("time.time") as time_mock:
        time_mock.side_effect = lambda: 123
        mlflow_client = tracking.MlflowClient()
        run_id = active_run.info.run_id

        mlflow_client.log_metric(run_id=run_id, key="name_1", value=25)
        mlflow_client.log_metric(run_id=run_id, key="name_2", value=-3)
        mlflow_client.log_metric(run_id=run_id, key="name_1", value=30)
        mlflow_client.log_metric(run_id=run_id, key="name_1", value=40)

    metric_history_name1 = mlflow_client.get_metric_history(run_id, "name_1")
    assert {(m.value, m.timestamp) for m in metric_history_name1} == {
        (25, 123 * 1000),
        (30, 123 * 1000),
        (40, 123 * 1000),
    }

    metric_history_name2 = mlflow_client.get_metric_history(run_id, "name_2")
    assert {(m.value, m.timestamp) for m in metric_history_name2} == {(-3, 123 * 1000)}


@pytest.mark.parametrize("step_kwarg", [None, -10, 5])
def test_log_metrics_uses_common_timestamp_and_step_per_invocation(step_kwarg):
    expected_metrics = {"name_1": 30, "name_2": -3, "nested/nested/name": 40}
    with start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.log_metrics(expected_metrics, step=step_kwarg)
    finished_run = tracking.MlflowClient().get_run(run_id)
    # Validate metric key/values match what we expect, and that all metrics have the same timestamp
    assert len(finished_run.data.metrics) == len(expected_metrics)
    for key, value in finished_run.data.metrics.items():
        assert expected_metrics[key] == value
    common_timestamp = finished_run.data._metric_objs[0].timestamp
    expected_step = step_kwarg if step_kwarg is not None else 0
    for metric_obj in finished_run.data._metric_objs:
        assert metric_obj.timestamp == common_timestamp
        assert metric_obj.step == expected_step


@pytest.fixture
def get_store_mock():
    with mock.patch("mlflow.store.file_store.FileStore.log_batch") as _get_store_mock:
        yield _get_store_mock


def test_set_tags():
    exact_expected_tags = {"name_1": "c", "name_2": "b", "nested/nested/name": 5}
    approx_expected_tags = {
        MLFLOW_USER,
        MLFLOW_SOURCE_NAME,
        MLFLOW_SOURCE_TYPE,
        MLFLOW_RUN_NAME,
    }
    with start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.set_tags(exact_expected_tags)
    finished_run = tracking.MlflowClient().get_run(run_id)
    # Validate tags
    assert len(finished_run.data.tags) == len(exact_expected_tags) + len(approx_expected_tags)
    for tag_key, tag_val in finished_run.data.tags.items():
        if tag_key in approx_expected_tags:
            pass
        else:
            assert str(exact_expected_tags[tag_key]) == tag_val


def test_log_metric_validation():
    with start_run() as active_run:
        run_id = active_run.info.run_id
        with pytest.raises(
            MlflowException,
            match="Invalid value \"apple\" for parameter 'value' supplied",
        ) as e:
            mlflow.log_metric("name_1", "apple")
    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    finished_run = tracking.MlflowClient().get_run(run_id)
    assert len(finished_run.data.metrics) == 0


def test_log_param():
    with start_run() as active_run:
        run_id = active_run.info.run_id
        assert mlflow.log_param("name_1", "a") == "a"
        assert mlflow.log_param("name_2", "b") == "b"
        assert mlflow.log_param("nested/nested/name", 5) == 5
    finished_run = tracking.MlflowClient().get_run(run_id)
    # Validate params
    assert finished_run.data.params == {
        "name_1": "a",
        "name_2": "b",
        "nested/nested/name": "5",
    }


def test_log_params():
    expected_params = {"name_1": "c", "name_2": "b", "nested/nested/name": 5}
    with start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.log_params(expected_params)
    finished_run = tracking.MlflowClient().get_run(run_id)
    # Validate params
    assert finished_run.data.params == {
        "name_1": "c",
        "name_2": "b",
        "nested/nested/name": "5",
    }


def test_log_params_duplicate_keys_raises():
    params = {"a": "1", "b": "2"}
    with start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.log_params(params)
        with pytest.raises(
            expected_exception=MlflowException,
            match=r"Changing param values is not allowed. Param with key=",
        ) as e:
            mlflow.log_param("a", "3")
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    finished_run = tracking.MlflowClient().get_run(run_id)
    assert finished_run.data.params == params


@pytest.mark.skipif(is_windows(), reason="Windows do not support colon in params and metrics")
def test_param_metric_with_colon():
    with start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.log_param("a:b", 3)
        mlflow.log_metric("c:d", 4)
    finished_run = tracking.MlflowClient().get_run(run_id)

    # Validate param
    assert len(finished_run.data.params) == 1
    assert finished_run.data.params == {"a:b": "3"}

    # Validate metric
    assert len(finished_run.data.metrics) == 1
    assert finished_run.data.metrics["c:d"] == 4


def test_log_batch_duplicate_entries_raises():
    with start_run() as active_run:
        run_id = active_run.info.run_id
        with pytest.raises(
            MlflowException, match=r"Duplicate parameter keys have been submitted."
        ) as e:
            tracking.MlflowClient().log_batch(
                run_id=run_id, params=[Param("a", "1"), Param("a", "2")]
            )
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_log_batch_validates_entity_names_and_values():
    with start_run() as active_run:
        run_id = active_run.info.run_id

        metrics = [Metric(key="../bad/metric/name", value=0.3, timestamp=3, step=0)]
        with pytest.raises(
            MlflowException,
            match=r"Invalid value \"../bad/metric/name\" for parameter \'metrics\[0\].name\'",
        ) as e:
            tracking.MlflowClient().log_batch(run_id, metrics=metrics)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        metrics = [Metric(key="ok-name", value="non-numerical-value", timestamp=3, step=0)]
        with pytest.raises(
            MlflowException,
            match=r"Invalid value \"non-numerical-value\" "
            + r"for parameter \'metrics\[0\].value\' supplied",
        ) as e:
            tracking.MlflowClient().log_batch(run_id, metrics=metrics)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        metrics = [Metric(key="ok-name", value=0.3, timestamp="non-numerical-timestamp", step=0)]
        with pytest.raises(
            MlflowException,
            match=r"Invalid value \"non-numerical-timestamp\" for "
            + r"parameter \'metrics\[0\].timestamp\' supplied",
        ) as e:
            tracking.MlflowClient().log_batch(run_id, metrics=metrics)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        params = [Param(key="../bad/param/name", value="my-val")]
        with pytest.raises(
            MlflowException,
            match=r"Invalid value \"../bad/param/name\" for parameter \'params\[0\].key\' supplied",
        ) as e:
            tracking.MlflowClient().log_batch(run_id, params=params)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        tags = [Param(key="../bad/tag/name", value="my-val")]
        with pytest.raises(
            MlflowException,
            match=r"Invalid value \"../bad/tag/name\" for parameter \'tags\[0\].key\' supplied",
        ) as e:
            tracking.MlflowClient().log_batch(run_id, tags=tags)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        metrics = [Metric(key=None, value=42.0, timestamp=4, step=1)]
        with pytest.raises(
            MlflowException,
            match="Metric name cannot be None. A key name must be provided.",
        ) as e:
            tracking.MlflowClient().log_batch(run_id, metrics=metrics)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_log_artifact_with_dirs(tmp_path):
    # Test log artifact with a directory
    art_dir = tmp_path / "parent"
    art_dir.mkdir()
    file0 = art_dir.joinpath("file0")
    file0.write_text("something")
    file1 = art_dir.joinpath("file1")
    file1.write_text("something")
    sub_dir = art_dir / "child"
    sub_dir.mkdir()
    with start_run():
        artifact_uri = mlflow.get_artifact_uri()
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        mlflow.log_artifact(str(art_dir))
        base = os.path.basename(str(art_dir))
        assert os.listdir(run_artifact_dir) == [base]
        assert set(os.listdir(os.path.join(run_artifact_dir, base))) == {
            "child",
            "file0",
            "file1",
        }
        with open(os.path.join(run_artifact_dir, base, "file0")) as f:
            assert f.read() == "something"
    # Test log artifact with directory and specified parent folder

    art_dir = tmp_path / "dir"
    art_dir.mkdir()
    with start_run():
        artifact_uri = mlflow.get_artifact_uri()
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        mlflow.log_artifact(str(art_dir), "some_parent")
        assert os.listdir(run_artifact_dir) == [os.path.basename("some_parent")]
        assert os.listdir(os.path.join(run_artifact_dir, "some_parent")) == [
            os.path.basename(str(art_dir))
        ]
    sub_dir = art_dir.joinpath("another_dir")
    sub_dir.mkdir()
    with start_run():
        artifact_uri = mlflow.get_artifact_uri()
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        mlflow.log_artifact(str(art_dir), "parent/and_child")
        assert os.listdir(os.path.join(run_artifact_dir, "parent", "and_child")) == [
            os.path.basename(str(art_dir))
        ]
        assert set(
            os.listdir(
                os.path.join(
                    run_artifact_dir,
                    "parent",
                    "and_child",
                    os.path.basename(str(art_dir)),
                )
            )
        ) == {os.path.basename(str(sub_dir))}


def test_log_artifact(tmp_path):
    # Create artifacts
    artifact_dir = tmp_path.joinpath("artifact_dir")
    artifact_dir.mkdir()
    path0 = artifact_dir.joinpath("file0")
    path1 = artifact_dir.joinpath("file1")
    path0.write_text("0")
    path1.write_text("1")
    # Log an artifact, verify it exists in the directory returned by get_artifact_uri
    # after the run finishes
    artifact_parent_dirs = ["some_parent_dir", None]
    for parent_dir in artifact_parent_dirs:
        with start_run():
            artifact_uri = mlflow.get_artifact_uri()
            run_artifact_dir = local_file_uri_to_path(artifact_uri)
            mlflow.log_artifact(path0, parent_dir)
        expected_dir = (
            os.path.join(run_artifact_dir, parent_dir)
            if parent_dir is not None
            else run_artifact_dir
        )
        assert os.listdir(expected_dir) == [os.path.basename(path0)]
        logged_artifact_path = os.path.join(expected_dir, path0)
        assert filecmp.cmp(logged_artifact_path, path0, shallow=False)
    # Log multiple artifacts, verify they exist in the directory returned by get_artifact_uri
    for parent_dir in artifact_parent_dirs:
        with start_run():
            artifact_uri = mlflow.get_artifact_uri()
            run_artifact_dir = local_file_uri_to_path(artifact_uri)

            mlflow.log_artifacts(artifact_dir, parent_dir)
        # Check that the logged artifacts match
        expected_artifact_output_dir = (
            os.path.join(run_artifact_dir, parent_dir)
            if parent_dir is not None
            else run_artifact_dir
        )
        dir_comparison = filecmp.dircmp(artifact_dir, expected_artifact_output_dir)
        assert len(dir_comparison.left_only) == 0
        assert len(dir_comparison.right_only) == 0
        assert len(dir_comparison.diff_files) == 0
        assert len(dir_comparison.funny_files) == 0


@pytest.mark.parametrize("subdir", [None, ".", "dir", "dir1/dir2", "dir/.."])
def test_log_text(subdir):
    filename = "file.txt"
    text = "a"
    artifact_file = filename if subdir is None else posixpath.join(subdir, filename)

    with mlflow.start_run():
        mlflow.log_text(text, artifact_file)

        artifact_path = None if subdir is None else posixpath.normpath(subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]

        filepath = os.path.join(run_artifact_dir, filename)
        with open(filepath) as f:
            assert f.read() == text


@pytest.mark.parametrize("subdir", [None, ".", "dir", "dir1/dir2", "dir/.."])
@pytest.mark.parametrize("extension", [".json", ".yml", ".yaml", ".txt", ""])
def test_log_dict(subdir, extension):
    dictionary = {"k": "v"}
    filename = "data" + extension
    artifact_file = filename if subdir is None else posixpath.join(subdir, filename)

    with mlflow.start_run():
        mlflow.log_dict(dictionary, artifact_file)

        artifact_path = None if subdir is None else posixpath.normpath(subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]

        filepath = os.path.join(run_artifact_dir, filename)
        extension = os.path.splitext(filename)[1]
        with open(filepath) as f:
            loaded = (
                # Specify `Loader` to suppress the following deprecation warning:
                # https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
                yaml.load(f, Loader=yaml.SafeLoader)
                if (extension in [".yml", ".yaml"])
                else json.load(f)
            )
            assert loaded == dictionary


def test_with_startrun():
    run_id = None
    t0 = get_current_time_millis()
    with mlflow.start_run() as active_run:
        assert mlflow.active_run() == active_run
        run_id = active_run.info.run_id
    t1 = get_current_time_millis()
    run_info = mlflow.tracking._get_store().get_run(run_id).info
    assert run_info.status == "FINISHED"
    assert t0 <= run_info.end_time
    assert run_info.end_time <= t1
    assert mlflow.active_run() is None


def test_parent_create_run(monkeypatch):
    with mlflow.start_run() as parent_run:
        parent_run_id = parent_run.info.run_id
    monkeypatch.setenv(MLFLOW_RUN_ID.name, parent_run_id)
    with mlflow.start_run() as parent_run:
        assert parent_run.info.run_id == parent_run_id
        with pytest.raises(Exception, match="To start a nested run"):
            mlflow.start_run()
        with mlflow.start_run(nested=True) as child_run:
            assert child_run.info.run_id != parent_run_id
            with mlflow.start_run(nested=True) as grand_child_run:
                pass

    def verify_has_parent_id_tag(child_id, expected_parent_id):
        tags = tracking.MlflowClient().get_run(child_id).data.tags
        assert tags[MLFLOW_PARENT_RUN_ID] == expected_parent_id

    verify_has_parent_id_tag(child_run.info.run_id, parent_run.info.run_id)
    verify_has_parent_id_tag(grand_child_run.info.run_id, child_run.info.run_id)
    assert mlflow.active_run() is None


def test_start_deleted_run():
    run_id = None
    with mlflow.start_run() as active_run:
        run_id = active_run.info.run_id
    tracking.MlflowClient().delete_run(run_id)
    with pytest.raises(MlflowException, match="because it is in the deleted state."):
        with mlflow.start_run(run_id=run_id):
            pass
    assert mlflow.active_run() is None


@pytest.mark.usefixtures("reset_active_experiment")
def test_start_run_exp_id_0():
    mlflow.set_experiment("some-experiment")
    # Create a run and verify that the current active experiment is the one we just set
    with mlflow.start_run() as active_run:
        exp_id = active_run.info.experiment_id
        assert exp_id != FileStore.DEFAULT_EXPERIMENT_ID
        assert MlflowClient().get_experiment(exp_id).name == "some-experiment"
    # Set experiment ID to 0 when creating a run, verify that the specified experiment ID is honored
    with mlflow.start_run(experiment_id=0) as active_run:
        assert active_run.info.experiment_id == FileStore.DEFAULT_EXPERIMENT_ID


def test_get_artifact_uri_with_artifact_path_unspecified_returns_artifact_root_dir():
    with mlflow.start_run() as active_run:
        assert mlflow.get_artifact_uri(artifact_path=None) == active_run.info.artifact_uri


def test_get_artifact_uri_uses_currently_active_run_id():
    artifact_path = "artifact"
    with mlflow.start_run() as active_run:
        assert mlflow.get_artifact_uri(
            artifact_path=artifact_path
        ) == tracking.artifact_utils.get_artifact_uri(
            run_id=active_run.info.run_id, artifact_path=artifact_path
        )


def _assert_get_artifact_uri_appends_to_uri_path_component_correctly(
    artifact_location, expected_uri_format
):
    client = MlflowClient()
    client.create_experiment("get-artifact-uri-test", artifact_location=artifact_location)
    mlflow.set_experiment("get-artifact-uri-test")
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        for artifact_path in ["path/to/artifact", "/artifact/path", "arty.txt"]:
            artifact_uri = mlflow.get_artifact_uri(artifact_path)
            assert artifact_uri == tracking.artifact_utils.get_artifact_uri(run_id, artifact_path)
            assert artifact_uri == expected_uri_format.format(
                run_id=run_id,
                path=artifact_path.lstrip("/"),
                drive=pathlib.Path.cwd().drive,
            )


@pytest.mark.parametrize(
    ("artifact_location", "expected_uri_format"),
    [
        (
            "mysql://user:password@host:port/dbname?driver=mydriver",
            "mysql://user:password@host:port/dbname/{run_id}/artifacts/{path}?driver=mydriver",
        ),
        (
            "mysql+driver://user:pass@host:port/dbname/subpath/#fragment",
            "mysql+driver://user:pass@host:port/dbname/subpath/{run_id}/artifacts/{path}#fragment",
        ),
        (
            "s3://bucketname/rootpath",
            "s3://bucketname/rootpath/{run_id}/artifacts/{path}",
        ),
    ],
)
def test_get_artifact_uri_appends_to_uri_path_component_correctly(
    artifact_location, expected_uri_format
):
    _assert_get_artifact_uri_appends_to_uri_path_component_correctly(
        artifact_location, expected_uri_format
    )


@pytest.mark.skipif(not is_windows(), reason="This test only passes on Windows")
def test_get_artifact_uri_appends_to_local_path_component_correctly_on_windows():
    _assert_get_artifact_uri_appends_to_uri_path_component_correctly(
        "/dirname/rootpa#th?",
        "file:///{drive}/dirname/rootpa/{run_id}/artifacts/{path}#th?",
    )


@pytest.mark.skipif(is_windows(), reason="This test fails on Windows")
def test_get_artifact_uri_appends_to_local_path_component_correctly():
    _assert_get_artifact_uri_appends_to_uri_path_component_correctly(
        "/dirname/rootpa#th?", "{drive}/dirname/rootpa#th?/{run_id}/artifacts/{path}"
    )


@pytest.mark.usefixtures("reset_active_experiment")
def test_search_runs():
    mlflow.set_experiment("exp-for-search")
    # Create a run and verify that the current active experiment is the one we just set
    logged_runs = {}
    with mlflow.start_run() as active_run:
        logged_runs["first"] = active_run.info.run_id
        mlflow.log_metric("m1", 0.001)
        mlflow.log_metric("m2", 0.002)
        mlflow.log_metric("m1", 0.002)
        mlflow.log_param("p1", "a")
        mlflow.set_tag("t1", "first-tag-val")
    with mlflow.start_run() as active_run:
        logged_runs["second"] = active_run.info.run_id
        mlflow.log_metric("m1", 0.008)
        mlflow.log_param("p2", "aa")
        mlflow.set_tag("t2", "second-tag-val")

    def verify_runs(runs, expected_set):
        assert {r.info.run_id for r in runs} == {logged_runs[r] for r in expected_set}

    experiment_id = MlflowClient().get_experiment_by_name("exp-for-search").experiment_id

    # 2 runs in this experiment
    assert len(MlflowClient().search_runs([experiment_id], run_view_type=ViewType.ACTIVE_ONLY)) == 2

    # 2 runs that have metric "m1" > 0.001
    runs = MlflowClient().search_runs([experiment_id], "metrics.m1 > 0.0001")
    verify_runs(runs, ["first", "second"])

    # 1 run with has metric "m1" > 0.002
    runs = MlflowClient().search_runs([experiment_id], "metrics.m1 > 0.002")
    verify_runs(runs, ["second"])

    # no runs with metric "m1" > 0.1
    runs = MlflowClient().search_runs([experiment_id], "metrics.m1 > 0.1")
    verify_runs(runs, [])

    # 1 run with metric "m2" > 0
    runs = MlflowClient().search_runs([experiment_id], "metrics.m2 > 0")
    verify_runs(runs, ["first"])

    # 1 run each with param "p1" and "p2"
    runs = MlflowClient().search_runs([experiment_id], "params.p1 = 'a'", ViewType.ALL)
    verify_runs(runs, ["first"])
    runs = MlflowClient().search_runs([experiment_id], "params.p2 != 'a'", ViewType.ALL)
    verify_runs(runs, ["second"])
    runs = MlflowClient().search_runs([experiment_id], "params.p2 = 'aa'", ViewType.ALL)
    verify_runs(runs, ["second"])

    # 1 run each with tag "t1" and "t2"
    runs = MlflowClient().search_runs([experiment_id], "tags.t1 = 'first-tag-val'", ViewType.ALL)
    verify_runs(runs, ["first"])
    runs = MlflowClient().search_runs([experiment_id], "tags.t2 != 'qwerty'", ViewType.ALL)
    verify_runs(runs, ["second"])
    runs = MlflowClient().search_runs([experiment_id], "tags.t2 = 'second-tag-val'", ViewType.ALL)
    verify_runs(runs, ["second"])

    # delete "first" run
    MlflowClient().delete_run(logged_runs["first"])
    runs = MlflowClient().search_runs([experiment_id], "params.p1 = 'a'", ViewType.ALL)
    verify_runs(runs, ["first"])

    runs = MlflowClient().search_runs([experiment_id], "params.p1 = 'a'", ViewType.DELETED_ONLY)
    verify_runs(runs, ["first"])

    runs = MlflowClient().search_runs([experiment_id], "params.p1 = 'a'", ViewType.ACTIVE_ONLY)
    verify_runs(runs, [])


@pytest.mark.usefixtures("reset_active_experiment")
def test_search_runs_multiple_experiments():
    experiment_ids = [mlflow.create_experiment(f"exp__{exp_id}") for exp_id in range(1, 4)]
    for eid in experiment_ids:
        with mlflow.start_run(experiment_id=eid):
            mlflow.log_metric("m0", 1)
            mlflow.log_metric(f"m_{eid}", 2)

    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m0 > 0", ViewType.ALL)) == 3

    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m_1 > 0", ViewType.ALL)) == 1
    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m_2 = 2", ViewType.ALL)) == 1
    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m_3 < 4", ViewType.ALL)) == 1


def read_data(artifact_path):
    import pandas as pd

    if artifact_path.endswith(".json"):
        return pd.read_json(artifact_path, orient="split")
    if artifact_path.endswith(".parquet"):
        return pd.read_parquet(artifact_path)
    raise ValueError(f"Unsupported file type in {artifact_path}. Expected .json or .parquet")


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny client does not support the np or pandas dependencies",
)
@pytest.mark.parametrize("file_type", ["json", "parquet"])
def test_log_table(file_type):
    import pandas as pd

    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }
    artifact_file = f"qabot_eval_results.{file_type}"
    TAG_NAME = "mlflow.loggedArtifacts"
    run_id = None

    with pytest.raises(
        MlflowException, match="data must be a pandas.DataFrame or a dictionary"
    ) as e:
        with mlflow.start_run() as run:
            # Log the incorrect data format as a table
            mlflow.log_table(data="incorrect-data-format", artifact_file=artifact_file)
    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    with mlflow.start_run() as run:
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file=artifact_file)
        run_id = run.info.run_id

    run = mlflow.get_run(run_id)
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_file)
    table_data = read_data(artifact_path)
    assert table_data.shape[0] == 2
    assert table_data.shape[1] == 3

    # Get the current value of the tag
    current_tag_value = json.loads(run.data.tags.get(TAG_NAME, "[]"))
    assert {"path": artifact_file, "type": "table"} in current_tag_value
    assert len(current_tag_value) == 1

    table_df = pd.DataFrame.from_dict(table_dict)
    with mlflow.start_run(run_id=run_id):
        # Log the dataframe as a table
        mlflow.log_table(data=table_df, artifact_file=artifact_file)

    run = mlflow.get_run(run_id)
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_file)
    table_data = read_data(artifact_path)
    assert table_data.shape[0] == 4
    assert table_data.shape[1] == 3
    # Get the current value of the tag
    current_tag_value = json.loads(run.data.tags.get(TAG_NAME, "[]"))
    assert {"path": artifact_file, "type": "table"} in current_tag_value
    assert len(current_tag_value) == 1

    artifact_file_new = f"qabot_eval_results_new.{file_type}"
    with mlflow.start_run(run_id=run_id):
        # Log the dataframe as a table to new artifact file
        mlflow.log_table(data=table_df, artifact_file=artifact_file_new)

    run = mlflow.get_run(run_id)
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_file_new
    )
    table_data = read_data(artifact_path)
    assert table_data.shape[0] == 2
    assert table_data.shape[1] == 3
    # Get the current value of the tag
    current_tag_value = json.loads(run.data.tags.get(TAG_NAME, "[]"))
    assert {"path": artifact_file_new, "type": "table"} in current_tag_value
    assert len(current_tag_value) == 2


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny client does not support the np or pandas dependencies",
)
@pytest.mark.parametrize("file_type", ["json", "parquet"])
def test_log_table_with_subdirectory(file_type):
    import pandas as pd

    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }
    artifact_file = f"dir/foo.{file_type}"
    TAG_NAME = "mlflow.loggedArtifacts"
    run_id = None

    with mlflow.start_run() as run:
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file=artifact_file)
        run_id = run.info.run_id

    run = mlflow.get_run(run_id)
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_file)
    table_data = read_data(artifact_path)
    assert table_data.shape[0] == 2
    assert table_data.shape[1] == 3

    # Get the current value of the tag
    current_tag_value = json.loads(run.data.tags.get(TAG_NAME, "[]"))
    assert {"path": artifact_file, "type": "table"} in current_tag_value
    assert len(current_tag_value) == 1

    table_df = pd.DataFrame.from_dict(table_dict)
    with mlflow.start_run(run_id=run_id):
        # Log the dataframe as a table
        mlflow.log_table(data=table_df, artifact_file=artifact_file)

    run = mlflow.get_run(run_id)
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_file)
    table_data = read_data(artifact_path)
    assert table_data.shape[0] == 4
    assert table_data.shape[1] == 3
    # Get the current value of the tag
    current_tag_value = json.loads(run.data.tags.get(TAG_NAME, "[]"))
    assert {"path": artifact_file, "type": "table"} in current_tag_value
    assert len(current_tag_value) == 1


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny client does not support the np or pandas dependencies",
)
@pytest.mark.parametrize("file_type", ["json", "parquet"])
def test_load_table(file_type):
    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "toxicity": [0.0, 0.0],
    }
    artifact_file = f"qabot_eval_results.{file_type}"
    artifact_file_2 = f"qabot_eval_results_2.{file_type}"
    run_id_2 = None

    with mlflow.start_run() as run:
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file=artifact_file)
        mlflow.log_table(data=table_dict, artifact_file=artifact_file_2)

    with mlflow.start_run() as run:
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file=artifact_file)
        run_id_2 = run.info.run_id

    with mlflow.start_run() as run:
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file=artifact_file)
        run_id_3 = run.info.run_id

    extra_columns = ["run_id", "tags.mlflow.loggedArtifacts"]

    # test 1: load table with extra columns
    output_df = mlflow.load_table(artifact_file=artifact_file, extra_columns=extra_columns)

    assert output_df.shape[0] == 6
    assert output_df.shape[1] == 5
    assert output_df["run_id"].nunique() == 3
    assert output_df["tags.mlflow.loggedArtifacts"].nunique() == 2

    # test 2: load table with extra columns and single run_id
    output_df = mlflow.load_table(
        artifact_file=artifact_file, run_ids=[run_id_2], extra_columns=extra_columns
    )

    assert output_df.shape[0] == 2
    assert output_df.shape[1] == 5
    assert output_df["run_id"].nunique() == 1
    assert output_df["tags.mlflow.loggedArtifacts"].nunique() == 1

    # test 3: load table with extra columns and multiple run_ids
    output_df = mlflow.load_table(
        artifact_file=artifact_file,
        run_ids=[run_id_2, run_id_3],
        extra_columns=extra_columns,
    )

    assert output_df.shape[0] == 4
    assert output_df.shape[1] == 5
    assert output_df["run_id"].nunique() == 2
    assert output_df["tags.mlflow.loggedArtifacts"].nunique() == 1

    # test 4: load table with no extra columns and run_ids specified but different artifact file
    output_df = mlflow.load_table(artifact_file=artifact_file_2)
    import pandas as pd

    pd.testing.assert_frame_equal(output_df, pd.DataFrame(table_dict), check_dtype=False)

    # test 5: load table with no extra columns and run_ids specified
    output_df = mlflow.load_table(artifact_file=artifact_file)

    assert output_df.shape[0] == 6
    assert output_df.shape[1] == 3

    # test 6: load table with no matching results found. Error case
    with pytest.raises(
        MlflowException, match="No runs found with the corresponding table artifact"
    ):
        mlflow.load_table(artifact_file=f"error_case.{file_type}")

    # test 7: load table with no matching extra_column found. Error case
    with pytest.raises(KeyError, match="error_column"):
        mlflow.load_table(artifact_file=artifact_file, extra_columns=["error_column"])


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny client does not support the np or pandas dependencies",
)
@pytest.mark.parametrize("file_type", ["json", "parquet"])
def test_log_table_with_datetime_columns(file_type):
    import pandas as pd

    start_time = str(datetime.now(timezone.utc))
    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "start_time": [start_time, start_time],
    }
    artifact_file = f"test_time.{file_type}"

    with mlflow.start_run() as run:
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file=artifact_file)
        run_id = run.info.run_id

    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_file)
    if file_type == "parquet":
        table_data = pd.read_parquet(artifact_path)
    else:
        table_data = pd.read_json(artifact_path, orient="split", convert_dates=False)
    assert table_data["start_time"][0] == start_time

    # append the same table to the same artifact file
    mlflow.log_table(data=table_dict, artifact_file=artifact_file, run_id=run_id)
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_file)
    if file_type == "parquet":
        df = pd.read_parquet(artifact_path)
    else:
        df = pd.read_json(artifact_path, orient="split", convert_dates=False)
    assert df["start_time"][2] == start_time


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny client does not support the np or pandas dependencies",
)
@pytest.mark.parametrize("file_type", ["json", "parquet"])
def test_log_table_with_image_columns(file_type):
    import numpy as np
    from PIL import Image

    image = mlflow.Image([[1, 2, 3]])
    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "image": [image, image],
    }
    artifact_file = f"test_time.{file_type}"

    with mlflow.start_run() as run:
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file=artifact_file)
        run_id = run.info.run_id

    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_file)
    table_data = read_data(artifact_path)
    assert table_data["image"][0]["type"] == "image"
    image_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=table_data["image"][0]["filepath"]
    )
    image2 = Image.open(image_path)
    assert np.abs(image.to_array() - np.array(image2)).sum() == 0

    # append the same table to the same artifact file
    mlflow.log_table(data=table_dict, artifact_file=artifact_file, run_id=run_id)
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_file)
    df = read_data(artifact_path)
    assert df["image"][2]["type"] == "image"


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny client does not support the np or pandas dependencies",
)
@pytest.mark.parametrize("file_type", ["json", "parquet"])
def test_log_table_with_pil_image_columns(file_type):
    import numpy as np
    from PIL import Image

    image = Image.fromarray(np.array([[1.0, 2.0, 3.0]]))
    image = image.convert("RGB")

    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "image": [image, image],
    }
    artifact_file = f"test_time.{file_type}"

    with mlflow.start_run() as run:
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file=artifact_file)
        run_id = run.info.run_id

    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_file)
    table_data = read_data(artifact_path)
    assert table_data["image"][0]["type"] == "image"
    image_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=table_data["image"][0]["filepath"]
    )
    image2 = Image.open(image_path)
    assert np.abs(np.array(image) - np.array(image2)).sum() == 0

    # append the same table to the same artifact file
    mlflow.log_table(data=table_dict, artifact_file=artifact_file, run_id=run_id)
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_file)
    df = read_data(artifact_path)
    assert df["image"][2]["type"] == "image"


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny client does not support the np or pandas dependencies",
)
@pytest.mark.parametrize("file_type", ["json", "parquet"])
def test_log_table_with_invalid_image_columns(file_type):
    image = mlflow.Image([[1, 2, 3]])
    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "image": [image, "text"],
    }
    artifact_file = f"test_time.{file_type}"
    with pytest.raises(ValueError, match="Column `image` contains a mix of images and non-images"):
        with mlflow.start_run():
            # Log the dictionary as a table
            mlflow.log_table(data=table_dict, artifact_file=artifact_file)


@pytest.mark.skipif(
    "MLFLOW_SKINNY" in os.environ,
    reason="Skinny client does not support the np or pandas dependencies",
)
@pytest.mark.parametrize("file_type", ["json", "parquet"])
def test_log_table_with_valid_image_columns(file_type):
    class ImageObj:
        def __init__(self):
            self.size = (1, 1)

        def resize(self, size):
            return self

        def save(self, path):
            with open(path, "w+") as f:
                f.write("dummy data")

    image_obj = ImageObj()
    image = mlflow.Image([[1, 2, 3]])

    table_dict = {
        "inputs": ["What is MLflow?", "What is Databricks?"],
        "outputs": ["MLflow is ...", "Databricks is ..."],
        "image": [image, image_obj],
    }
    # No error should be raised
    artifact_file = f"test_time.{file_type}"
    with mlflow.start_run():
        # Log the dictionary as a table
        mlflow.log_table(data=table_dict, artifact_file=artifact_file)


def test_set_async_logging_threadpool_size():
    MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE.set(6)
    assert MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE.get() == 6

    with mlflow.start_run():
        mlflow.log_param("key", "val", synchronous=False)

    store = mlflow.tracking._get_store()
    async_queue = store._async_logging_queue
    assert async_queue._batch_logging_worker_threadpool._max_workers == 6
    mlflow.flush_async_logging()
    MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE.unset()
