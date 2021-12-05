from collections import namedtuple
import filecmp
import json
import os
import posixpath
import random
import tempfile
import time
import yaml
import re

import pytest
from unittest import mock

import mlflow
from mlflow import tracking
from mlflow.entities import RunStatus, LifecycleStage, Metric, Param, RunTag, ViewType
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.file_store import FileStore
from mlflow.protos.databricks_pb2 import ErrorCode, INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import start_run
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.mlflow_tags import (
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_USER,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
)
from mlflow.tracking.fluent import _RUN_ID_ENV_VAR

MockExperiment = namedtuple("MockExperiment", ["experiment_id", "lifecycle_stage"])


def test_create_experiment():
    with pytest.raises(MlflowException, match="Invalid experiment name"):
        mlflow.create_experiment(None)

    with pytest.raises(MlflowException, match="Invalid experiment name"):
        mlflow.create_experiment("")

    exp_id = mlflow.create_experiment("Some random experiment name %d" % random.randint(1, 1e6))
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
        MlflowException, match=re.escape(f"Invalid experiment name: {name}. Expects a string.")
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


def test_list_experiments():
    def _assert_exps(ids_to_lifecycle_stage, view_type_arg):
        result = set(
            [
                (exp.experiment_id, exp.lifecycle_stage)
                for exp in client.list_experiments(view_type=view_type_arg)
            ]
        )
        assert result == set([(exp_id, stage) for exp_id, stage in ids_to_lifecycle_stage.items()])

    experiment_id = mlflow.create_experiment("exp_1")
    assert experiment_id == "1"
    client = tracking.MlflowClient()
    _assert_exps({"0": LifecycleStage.ACTIVE, "1": LifecycleStage.ACTIVE}, ViewType.ACTIVE_ONLY)
    _assert_exps({"0": LifecycleStage.ACTIVE, "1": LifecycleStage.ACTIVE}, ViewType.ALL)
    _assert_exps({}, ViewType.DELETED_ONLY)
    client.delete_experiment(experiment_id)
    _assert_exps({"0": LifecycleStage.ACTIVE}, ViewType.ACTIVE_ONLY)
    _assert_exps({"0": LifecycleStage.ACTIVE, "1": LifecycleStage.DELETED}, ViewType.ALL)
    _assert_exps({"1": LifecycleStage.DELETED}, ViewType.DELETED_ONLY)


def test_list_experiments_paginated():
    experiments = []
    for i in range(10):
        experiments.append(mlflow.create_experiment("paginated_exp_" + str(i)))
    max_results = 5
    returned_experiments = []
    client = tracking.MlflowClient()
    result = client.list_experiments(max_results=max_results, page_token=None)
    assert len(result) == max_results
    returned_experiments.extend(result)
    while result.token:
        result = client.list_experiments(max_results=max_results, page_token=result.token)
        assert len(result) <= max_results
        returned_experiments.extend(result)
    assert result.token is None
    returned_exp_id_set = set([exp.experiment_id for exp in returned_experiments])
    assert set(experiments) - returned_exp_id_set == set()


def test_list_experiments_paginated_returns_in_correct_order():
    testnames = []
    for i in range(20):
        name = "paginated_exp_order_" + str(i)
        mlflow.create_experiment(name)
        testnames.append(name)

    client = tracking.MlflowClient()
    # test that pagination will return all valid results in sorted order
    # by name ascending
    result = client.list_experiments(max_results=3, page_token=None)
    assert result.token is not None
    assert [exp.name for exp in result[1:]] == testnames[0:2]

    result = client.list_experiments(max_results=4, page_token=result.token)
    assert result.token is not None
    assert [exp.name for exp in result] == testnames[2:6]

    result = client.list_experiments(max_results=6, page_token=result.token)
    assert result.token is not None
    assert [exp.name for exp in result] == testnames[6:12]

    result = client.list_experiments(max_results=8, page_token=result.token)
    # this page token should be none
    assert result.token is None
    assert [exp.name for exp in result] == testnames[12:]


def test_list_experiments_paginated_errors():
    client = tracking.MlflowClient()
    # test that providing a completely invalid page token throws
    with pytest.raises(MlflowException, match="Invalid page token") as exception_context:
        client.list_experiments(page_token="evilhax", max_results=20)
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    # test that providing too large of a max_results throws
    with pytest.raises(
        MlflowException, match="Invalid value for request parameter max_results"
    ) as exception_context:
        client.list_experiments(page_token=None, max_results=int(1e15))
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


@pytest.mark.usefixtures("reset_active_experiment")
def test_set_experiment_with_zero_id(reset_mock):
    reset_mock(
        MlflowClient,
        "get_experiment_by_name",
        mock.Mock(
            return_value=MockExperiment(experiment_id=0, lifecycle_stage=LifecycleStage.ACTIVE)
        ),
    )
    reset_mock(MlflowClient, "create_experiment", mock.Mock())

    mlflow.set_experiment("my_exp")

    MlflowClient.get_experiment_by_name.assert_called_once()
    MlflowClient.create_experiment.assert_not_called()


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
            second_run_id = second_run.info.run_id
            raise Exception("Failing run!")
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
    client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(run_id, "name_1")
    finished_run = client.get_run(run_id)
    assert len(history) == 2
    assert all(
        [
            m.timestamp >= finished_run.info.start_time
            and m.timestamp <= finished_run.info.end_time
            for m in history
        ]
    )


@pytest.mark.usefixtures("tmpdir")
def test_log_batch():
    expected_metrics = {"metric-key0": 1.0, "metric-key1": 4.0}
    expected_params = {"param-key0": "param-val0", "param-key1": "param-val1"}
    exact_expected_tags = {"tag-key0": "tag-val0", "tag-key1": "tag-val1"}
    approx_expected_tags = set([MLFLOW_USER, MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE])

    t = int(time.time())
    sorted_expected_metrics = sorted(expected_metrics.items(), key=lambda kv: kv[0])
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
    assert len(finished_run.data.metrics) == 2
    for key, value in finished_run.data.metrics.items():
        assert expected_metrics[key] == value
    metric_history0 = client.get_metric_history(run_id, "metric-key0")
    assert set([(m.value, m.timestamp, m.step) for m in metric_history0]) == set([(1.0, t, 0)])
    metric_history1 = client.get_metric_history(run_id, "metric-key1")
    assert set([(m.value, m.timestamp, m.step) for m in metric_history1]) == set([(4.0, t, 1)])

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
    assert set([(m.value, m.timestamp, m.step) for m in metric_history_name1]) == set(
        [(25, 123 * 1000, 0), (30, 123 * 1000, 5), (40, 123 * 1000, -2)]
    )
    metric_history_name2 = client.get_metric_history(run_id, "name_2")
    assert set([(m.value, m.timestamp, m.step) for m in metric_history_name2]) == set(
        [(-3, 123 * 1000, 0)]
    )


def test_log_metrics_uses_millisecond_timestamp_resolution_fluent():
    with start_run() as active_run, mock.patch("time.time") as time_mock:
        time_mock.side_effect = lambda: 123
        mlflow.log_metrics({"name_1": 25, "name_2": -3})
        mlflow.log_metrics({"name_1": 30})
        mlflow.log_metrics({"name_1": 40})
        run_id = active_run.info.run_id

    client = tracking.MlflowClient()
    metric_history_name1 = client.get_metric_history(run_id, "name_1")
    assert set([(m.value, m.timestamp) for m in metric_history_name1]) == set(
        [(25, 123 * 1000), (30, 123 * 1000), (40, 123 * 1000)]
    )
    metric_history_name2 = client.get_metric_history(run_id, "name_2")
    assert set([(m.value, m.timestamp) for m in metric_history_name2]) == set([(-3, 123 * 1000)])


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
    assert set([(m.value, m.timestamp) for m in metric_history_name1]) == set(
        [(25, 123 * 1000), (30, 123 * 1000), (40, 123 * 1000)]
    )
    metric_history_name2 = mlflow_client.get_metric_history(run_id, "name_2")
    assert set([(m.value, m.timestamp) for m in metric_history_name2]) == set([(-3, 123 * 1000)])


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
@pytest.mark.usefixtures("tmpdir")
def get_store_mock():
    with mock.patch("mlflow.store.file_store.FileStore.log_batch") as _get_store_mock:
        yield _get_store_mock


def test_set_tags():
    exact_expected_tags = {"name_1": "c", "name_2": "b", "nested/nested/name": 5}
    approx_expected_tags = set([MLFLOW_USER, MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE])
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
        with pytest.raises(MlflowException, match="Got invalid value apple for metric") as e:
            mlflow.log_metric("name_1", "apple")
    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    finished_run = tracking.MlflowClient().get_run(run_id)
    assert len(finished_run.data.metrics) == 0


def test_log_param():
    with start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.log_param("name_1", "a")
        mlflow.log_param("name_2", "b")
        mlflow.log_param("nested/nested/name", 5)
    finished_run = tracking.MlflowClient().get_run(run_id)
    # Validate params
    assert finished_run.data.params == {"name_1": "a", "name_2": "b", "nested/nested/name": "5"}


def test_log_params():
    expected_params = {"name_1": "c", "name_2": "b", "nested/nested/name": 5}
    with start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.log_params(expected_params)
    finished_run = tracking.MlflowClient().get_run(run_id)
    # Validate params
    assert finished_run.data.params == {"name_1": "c", "name_2": "b", "nested/nested/name": "5"}


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
        with pytest.raises(MlflowException, match="Invalid metric name") as e:
            tracking.MlflowClient().log_batch(run_id, metrics=metrics)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        metrics = [Metric(key="ok-name", value="non-numerical-value", timestamp=3, step=0)]
        with pytest.raises(MlflowException, match="Got invalid value") as e:
            tracking.MlflowClient().log_batch(run_id, metrics=metrics)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        metrics = [Metric(key="ok-name", value=0.3, timestamp="non-numerical-timestamp", step=0)]
        with pytest.raises(MlflowException, match="Got invalid timestamp") as e:
            tracking.MlflowClient().log_batch(run_id, metrics=metrics)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        params = [Param(key="../bad/param/name", value="my-val")]
        with pytest.raises(MlflowException, match="Invalid parameter name") as e:
            tracking.MlflowClient().log_batch(run_id, params=params)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

        tags = [Param(key="../bad/tag/name", value="my-val")]
        with pytest.raises(MlflowException, match="Invalid tag name") as e:
            tracking.MlflowClient().log_batch(run_id, tags=tags)
        assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_log_artifact_with_dirs(tmpdir):
    # Test log artifact with a directory
    art_dir = tmpdir.mkdir("parent")
    file0 = art_dir.join("file0")
    file0.write("something")
    file1 = art_dir.join("file1")
    file1.write("something")
    sub_dir = art_dir.mkdir("child")
    with start_run():
        artifact_uri = mlflow.get_artifact_uri()
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        mlflow.log_artifact(str(art_dir))
        base = os.path.basename(str(art_dir))
        assert os.listdir(run_artifact_dir) == [base]
        assert set(os.listdir(os.path.join(run_artifact_dir, base))) == {"child", "file0", "file1"}
        with open(os.path.join(run_artifact_dir, base, "file0")) as f:
            assert f.read() == "something"
    # Test log artifact with directory and specified parent folder
    art_dir = tmpdir.mkdir("dir")
    with start_run():
        artifact_uri = mlflow.get_artifact_uri()
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        mlflow.log_artifact(str(art_dir), "some_parent")
        assert os.listdir(run_artifact_dir) == [os.path.basename("some_parent")]
        assert os.listdir(os.path.join(run_artifact_dir, "some_parent")) == [
            os.path.basename(str(art_dir))
        ]
    sub_dir = art_dir.mkdir("another_dir")
    with start_run():
        artifact_uri = mlflow.get_artifact_uri()
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        mlflow.log_artifact(str(art_dir), "parent/and_child")
        assert os.listdir(os.path.join(run_artifact_dir, "parent", "and_child")) == [
            os.path.basename(str(art_dir))
        ]
        assert os.listdir(
            os.path.join(run_artifact_dir, "parent", "and_child", os.path.basename(str(art_dir)))
        ) == [os.path.basename(str(sub_dir))]


def test_log_artifact():
    artifact_src_dir = tempfile.mkdtemp()
    # Create artifacts
    _, path0 = tempfile.mkstemp(dir=artifact_src_dir)
    _, path1 = tempfile.mkstemp(dir=artifact_src_dir)
    for i, path in enumerate([path0, path1]):
        with open(path, "w") as handle:
            handle.write("%s" % str(i))
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

            mlflow.log_artifacts(artifact_src_dir, parent_dir)
        # Check that the logged artifacts match
        expected_artifact_output_dir = (
            os.path.join(run_artifact_dir, parent_dir)
            if parent_dir is not None
            else run_artifact_dir
        )
        dir_comparison = filecmp.dircmp(artifact_src_dir, expected_artifact_output_dir)
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


@pytest.mark.large
@pytest.mark.parametrize("subdir", [None, ".", "dir", "dir1/dir2", "dir/.."])
def test_log_figure_matplotlib(subdir):
    import matplotlib.pyplot as plt

    filename = "figure.png"
    artifact_file = filename if subdir is None else posixpath.join(subdir, filename)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [2, 3])

    with mlflow.start_run():
        mlflow.log_figure(fig, artifact_file)
        plt.close(fig)

        artifact_path = None if subdir is None else posixpath.normpath(subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]


@pytest.mark.large
@pytest.mark.parametrize("subdir", [None, ".", "dir", "dir1/dir2", "dir/.."])
def test_log_figure_plotly(subdir):
    from plotly import graph_objects as go

    filename = "figure.html"
    artifact_file = filename if subdir is None else posixpath.join(subdir, filename)

    fig = go.Figure(go.Scatter(x=[0, 1], y=[2, 3]))

    with mlflow.start_run():
        mlflow.log_figure(fig, artifact_file)

        artifact_path = None if subdir is None else posixpath.normpath(subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]


@pytest.mark.large
def test_log_figure_raises_error_for_unsupported_figure_object_type():
    with mlflow.start_run(), pytest.raises(TypeError, match="Unsupported figure object type"):
        mlflow.log_figure("not_figure", "figure.png")


@pytest.mark.large
@pytest.mark.parametrize("subdir", [None, ".", "dir", "dir1/dir2", "dir/.."])
def test_log_image_numpy(subdir):
    import numpy as np
    from PIL import Image

    filename = "image.png"
    artifact_file = filename if subdir is None else posixpath.join(subdir, filename)

    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    with mlflow.start_run():
        mlflow.log_image(image, artifact_file)

        artifact_path = None if subdir is None else posixpath.normpath(subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]

        logged_path = os.path.join(run_artifact_dir, filename)
        loaded_image = np.asarray(Image.open(logged_path), dtype=np.uint8)
        np.testing.assert_array_equal(loaded_image, image)


@pytest.mark.large
@pytest.mark.parametrize("subdir", [None, ".", "dir", "dir1/dir2", "dir/.."])
def test_log_image_pillow(subdir):
    from PIL import Image
    from PIL import ImageChops

    filename = "image.png"
    artifact_file = filename if subdir is None else posixpath.join(subdir, filename)

    image = Image.new("RGB", (100, 100))

    with mlflow.start_run():
        mlflow.log_image(image, artifact_file)

        artifact_path = None if subdir is None else posixpath.normpath(subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]

        logged_path = os.path.join(run_artifact_dir, filename)
        loaded_image = Image.open(logged_path)
        # How to check Pillow image equality: https://stackoverflow.com/a/6204954/6943581
        assert ImageChops.difference(loaded_image, image).getbbox() is None


@pytest.mark.large
@pytest.mark.parametrize(
    "size",
    [
        (100, 100),  # Grayscale (2D)
        (100, 100, 1),  # Grayscale (3D)
        (100, 100, 3),  # RGB
        (100, 100, 4),  # RGBA
    ],
)
def test_log_image_numpy_shape(size):
    import numpy as np

    filename = "image.png"
    image = np.random.randint(0, 256, size=size, dtype=np.uint8)

    with mlflow.start_run():
        mlflow.log_image(image, filename)
        artifact_uri = mlflow.get_artifact_uri()
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]


@pytest.mark.large
@pytest.mark.parametrize(
    "dtype",
    [
        # Ref.: https://numpy.org/doc/stable/user/basics.types.html#array-types-and-conversions-between-types
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "bool",
    ],
)
def test_log_image_numpy_dtype(dtype):
    import numpy as np

    filename = "image.png"
    image = np.random.randint(0, 2, size=(100, 100, 3)).astype(np.dtype(dtype))

    with mlflow.start_run():
        mlflow.log_image(image, filename)
        artifact_uri = mlflow.get_artifact_uri()
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]


@pytest.mark.large
@pytest.mark.parametrize(
    "array",
    # 1 pixel images with out-of-range values
    [[[-1]], [[256]], [[-0.1]], [[1.1]]],
)
def test_log_image_numpy_emits_warning_for_out_of_range_values(array):
    import numpy as np

    image = np.array(array).astype(type(array[0][0]))

    with mlflow.start_run(), mock.patch("mlflow.tracking.client._logger.warning") as warn_mock:
        mlflow.log_image(image, "image.png")
        range_str = "[0, 255]" if isinstance(array[0][0], int) else "[0, 1]"
        msg = "Out-of-range values are detected. Clipping array (dtype: '{}') to {}".format(
            image.dtype, range_str
        )
        assert any(msg in args[0] for args in warn_mock.call_args_list)


@pytest.mark.large
def test_log_image_numpy_raises_exception_for_invalid_array_data_type():
    import numpy as np

    with mlflow.start_run(), pytest.raises(TypeError, match="Invalid array data type"):
        mlflow.log_image(np.tile("a", (1, 1, 3)), "image.png")


@pytest.mark.large
def test_log_image_numpy_raises_exception_for_invalid_array_shape():
    import numpy as np

    with mlflow.start_run(), pytest.raises(ValueError, match="`image` must be a 2D or 3D array"):
        mlflow.log_image(np.zeros((1,), dtype=np.uint8), "image.png")


@pytest.mark.large
def test_log_image_numpy_raises_exception_for_invalid_channel_length():
    import numpy as np

    with mlflow.start_run(), pytest.raises(ValueError, match="Invalid channel length"):
        mlflow.log_image(np.zeros((1, 1, 5), dtype=np.uint8), "image.png")


@pytest.mark.large
def test_log_image_raises_exception_for_unsupported_image_object_type():
    with mlflow.start_run(), pytest.raises(TypeError, match="Unsupported image object type"):
        mlflow.log_image("not_image", "image.png")


def test_with_startrun():
    run_id = None
    t0 = int(time.time() * 1000)
    with mlflow.start_run() as active_run:
        assert mlflow.active_run() == active_run
        run_id = active_run.info.run_id
    t1 = int(time.time() * 1000)
    run_info = mlflow.tracking._get_store().get_run(run_id).info
    assert run_info.status == "FINISHED"
    assert t0 <= run_info.end_time and run_info.end_time <= t1
    assert mlflow.active_run() is None


def test_parent_create_run():
    with mlflow.start_run() as parent_run:
        parent_run_id = parent_run.info.run_id
    os.environ[_RUN_ID_ENV_VAR] = parent_run_id
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


@pytest.mark.parametrize(
    "artifact_location, expected_uri_format",
    [
        (
            "mysql://user:password@host:port/dbname?driver=mydriver",
            "mysql://user:password@host:port/dbname/{run_id}/artifacts/{path}?driver=mydriver",
        ),
        (
            "mysql+driver://user:password@host:port/dbname/subpath/#fragment",
            "mysql+driver://user:password@host:port/dbname/subpath/{run_id}/artifacts/{path}#fragment",  # pylint: disable=line-too-long
        ),
        ("s3://bucketname/rootpath", "s3://bucketname/rootpath/{run_id}/artifacts/{path}"),
        ("/dirname/rootpa#th?", "/dirname/rootpa#th?/{run_id}/artifacts/{path}"),
    ],
)
def test_get_artifact_uri_appends_to_uri_path_component_correctly(
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
                run_id=run_id, path=artifact_path.lstrip("/")
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
        assert set([r.info.run_id for r in runs]) == set([logged_runs[r] for r in expected_set])

    experiment_id = MlflowClient().get_experiment_by_name("exp-for-search").experiment_id

    # 2 runs in this experiment
    assert len(MlflowClient().list_run_infos(experiment_id, ViewType.ACTIVE_ONLY)) == 2

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
    experiment_ids = [mlflow.create_experiment("exp__{}".format(exp_id)) for exp_id in range(1, 4)]
    for eid in experiment_ids:
        with mlflow.start_run(experiment_id=eid):
            mlflow.log_metric("m0", 1)
            mlflow.log_metric("m_{}".format(eid), 2)

    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m0 > 0", ViewType.ALL)) == 3

    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m_1 > 0", ViewType.ALL)) == 1
    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m_2 = 2", ViewType.ALL)) == 1
    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m_3 < 4", ViewType.ALL)) == 1
