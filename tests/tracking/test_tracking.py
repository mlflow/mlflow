from __future__ import print_function
import filecmp
import os
import random
import tempfile
import time

import attrdict
import mock
import pytest

import mlflow
from mlflow import tracking
from mlflow.entities import RunStatus, LifecycleStage, Metric, Param, RunTag, ViewType
from mlflow.exceptions import MlflowException
from mlflow.store.file_store import FileStore
from mlflow.protos.databricks_pb2 import ErrorCode, INVALID_PARAMETER_VALUE
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import start_run, end_run
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE

from tests.projects.utils import tracking_uri_mock


def test_create_experiment(tracking_uri_mock):
    with pytest.raises(TypeError):
        mlflow.create_experiment()

    with pytest.raises(Exception):
        mlflow.create_experiment(None)

    with pytest.raises(Exception):
        mlflow.create_experiment("")

    exp_id = mlflow.create_experiment(
        "Some random experiment name %d" % random.randint(1, 1e6))
    assert exp_id is not None


def test_create_experiment_with_duplicate_name(tracking_uri_mock):
    name = "popular_name"
    exp_id = mlflow.create_experiment(name)

    with pytest.raises(MlflowException):
        mlflow.create_experiment(name)

    tracking.MlflowClient().delete_experiment(exp_id)
    with pytest.raises(MlflowException):
        mlflow.create_experiment(name)


def test_create_experiments_with_bad_names():
    # None for name
    with pytest.raises(MlflowException) as e:
        mlflow.create_experiment(None)
        assert e.message.contains("Invalid experiment name: 'None'")

    # empty string name
    with pytest.raises(MlflowException) as e:
        mlflow.create_experiment("")
        assert e.message.contains("Invalid experiment name: ''")


@pytest.mark.parametrize("name", [123, 0, -1.2, [], ["A"], {1: 2}])
def test_create_experiments_with_bad_name_types(name):
    with pytest.raises(MlflowException) as e:
        mlflow.create_experiment(name)
        assert e.message.contains("Invalid experiment name: %s. Expects a string." % name)


def test_set_experiment(tracking_uri_mock, reset_active_experiment):
    with pytest.raises(TypeError):
        mlflow.set_experiment()

    with pytest.raises(Exception):
        mlflow.set_experiment(None)

    with pytest.raises(Exception):
        mlflow.set_experiment("")

    name = "random_exp"
    exp_id = mlflow.create_experiment(name)
    mlflow.set_experiment(name)
    with start_run() as run:
        assert run.info.experiment_id == exp_id

    another_name = "another_experiment"
    mlflow.set_experiment(another_name)
    exp_id2 = mlflow.tracking.MlflowClient().get_experiment_by_name(another_name)
    with start_run() as another_run:
        assert another_run.info.experiment_id == exp_id2.experiment_id


def test_set_experiment_with_deleted_experiment_name(tracking_uri_mock):
    name = "dead_exp"
    mlflow.set_experiment(name)
    with start_run() as run:
        exp_id = run.info.experiment_id

    tracking.MlflowClient().delete_experiment(exp_id)

    with pytest.raises(MlflowException):
        mlflow.set_experiment(name)


def test_list_experiments(tracking_uri_mock):
    def _assert_exps(ids_to_lifecycle_stage, view_type_arg):
        result = set([(exp.experiment_id, exp.lifecycle_stage)
                      for exp in client.list_experiments(view_type=view_type_arg)])
        assert result == set([(id, stage) for id, stage in ids_to_lifecycle_stage.items()])
    experiment_id = mlflow.create_experiment("exp_1")
    assert experiment_id == '1'
    client = tracking.MlflowClient()
    _assert_exps({'0': LifecycleStage.ACTIVE, '1': LifecycleStage.ACTIVE}, ViewType.ACTIVE_ONLY)
    _assert_exps({'0': LifecycleStage.ACTIVE, '1': LifecycleStage.ACTIVE}, ViewType.ALL)
    _assert_exps({}, ViewType.DELETED_ONLY)
    client.delete_experiment(experiment_id)
    _assert_exps({'0': LifecycleStage.ACTIVE}, ViewType.ACTIVE_ONLY)
    _assert_exps({'0': LifecycleStage.ACTIVE, '1': LifecycleStage.DELETED}, ViewType.ALL)
    _assert_exps({'1': LifecycleStage.DELETED}, ViewType.DELETED_ONLY)


def test_set_experiment_with_zero_id(reset_mock, reset_active_experiment):
    reset_mock(MlflowClient, "get_experiment_by_name",
               mock.Mock(return_value=attrdict.AttrDict(
                   experiment_id=0,
                   lifecycle_stage=LifecycleStage.ACTIVE)))
    reset_mock(MlflowClient, "create_experiment", mock.Mock())

    mlflow.set_experiment("my_exp")

    MlflowClient.get_experiment_by_name.assert_called_once()
    MlflowClient.create_experiment.assert_not_called()


def test_start_run_context_manager(tracking_uri_mock):
    with start_run() as first_run:
        first_uuid = first_run.info.run_id
        # Check that start_run() causes the run information to be persisted in the store
        persisted_run = tracking.MlflowClient().get_run(first_uuid)
        assert persisted_run is not None
        assert persisted_run.info == first_run.info
    finished_run = tracking.MlflowClient().get_run(first_uuid)
    assert finished_run.info.status == RunStatus.FINISHED
    # Launch a separate run that fails, verify the run status is FAILED and the run UUID is
    # different
    with pytest.raises(Exception):
        with start_run() as second_run:
            second_run_id = second_run.info.run_id
            raise Exception("Failing run!")
    assert second_run_id != first_uuid
    finished_run2 = tracking.MlflowClient().get_run(second_run_id)
    assert finished_run2.info.status == RunStatus.FAILED


def test_start_and_end_run(tracking_uri_mock):
    # Use the start_run() and end_run() APIs without a `with` block, verify they work.

    with start_run() as active_run:
        mlflow.log_metric("name_1", 25)
    finished_run = tracking.MlflowClient().get_run(active_run.info.run_id)
    # Validate metrics
    assert len(finished_run.data.metrics) == 1
    assert finished_run.data.metrics["name_1"] == 25


def test_metric_timestamp(tracking_uri_mock):
    with mlflow.start_run() as active_run:
        mlflow.log_metric("name_1", 25)
        mlflow.log_metric("name_1", 30)
        run_id = active_run.info.run_uuid
    # Check that metric timestamps are between run start and finish
    client = mlflow.tracking.MlflowClient()
    history = client.get_metric_history(run_id, "name_1")
    finished_run = client.get_run(run_id)
    assert len(history) == 2
    assert all([
        m.timestamp >= finished_run.info.start_time and m.timestamp <= finished_run.info.end_time
        for m in history
    ])


def test_log_batch(tracking_uri_mock, tmpdir):
    expected_metrics = {"metric-key0": 1.0, "metric-key1": 4.0}
    expected_params = {"param-key0": "param-val0", "param-key1": "param-val1"}
    exact_expected_tags = {"tag-key0": "tag-val0", "tag-key1": "tag-val1"}
    approx_expected_tags = set([MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE])

    t = int(time.time())
    sorted_expected_metrics = sorted(expected_metrics.items(), key=lambda kv: kv[0])
    metrics = [Metric(key=key, value=value, timestamp=t, step=i)
               for i, (key, value) in enumerate(sorted_expected_metrics)]
    params = [Param(key=key, value=value) for key, value in expected_params.items()]
    tags = [RunTag(key=key, value=value) for key, value in exact_expected_tags.items()]

    with start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.tracking.MlflowClient().log_batch(run_id=run_id, metrics=metrics, params=params,
                                                 tags=tags)
    client = tracking.MlflowClient()
    finished_run = client.get_run(run_id)
    # Validate metrics
    assert len(finished_run.data.metrics) == 2
    for key, value in finished_run.data.metrics.items():
        assert expected_metrics[key] == value
    metric_history0 = client.get_metric_history(run_id, "metric-key0")
    assert set([(m.value, m.timestamp, m.step) for m in metric_history0]) == set([
        (1.0, t, 0),
    ])
    metric_history1 = client.get_metric_history(run_id, "metric-key1")
    assert set([(m.value, m.timestamp, m.step) for m in metric_history1]) == set([
        (4.0, t, 1),
    ])

    # Validate tags (for automatically-set tags)
    assert len(finished_run.data.tags) == len(exact_expected_tags) + len(approx_expected_tags)
    for tag_key, tag_value in finished_run.data.tags.items():
        if tag_key in approx_expected_tags:
            pass
        else:
            assert exact_expected_tags[tag_key] == tag_value
    # Validate params
    assert finished_run.data.params == expected_params


def test_log_metric(tracking_uri_mock, tmpdir):
    with start_run() as active_run, mock.patch("time.time") as time_mock:
        time_mock.side_effect = range(300, 400)
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
    assert set([(m.value, m.timestamp, m.step) for m in metric_history_name1]) == set([
        (25, 300 * 1000, 0),
        (30, 302 * 1000, 5),
        (40, 303 * 1000, -2),
    ])
    metric_history_name2 = client.get_metric_history(run_id, "name_2")
    assert set([(m.value, m.timestamp, m.step) for m in metric_history_name2]) == set([
        (-3, 301 * 1000, 0),
    ])


@pytest.mark.parametrize("step_kwarg", [None, -10, 5])
def test_log_metrics(tracking_uri_mock, step_kwarg):
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
def get_store_mock(tmpdir):
    with mock.patch("mlflow.store.file_store.FileStore.log_batch") as _get_store_mock:
        yield _get_store_mock


def test_set_tags(tracking_uri_mock):
    exact_expected_tags = {"name_1": "c", "name_2": "b", "nested/nested/name": "5"}
    approx_expected_tags = set([MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE])
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
            assert exact_expected_tags[tag_key] == tag_val


def test_log_metric_validation(tracking_uri_mock):
    with start_run() as active_run:
        run_id = active_run.info.run_id
        with pytest.raises(MlflowException) as e:
            mlflow.log_metric("name_1", "apple")
    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    finished_run = tracking.MlflowClient().get_run(run_id)
    assert len(finished_run.data.metrics) == 0


def test_log_param(tracking_uri_mock):
    with start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.log_param("name_1", "a")
        mlflow.log_param("name_2", "b")
        mlflow.log_param("name_1", "c")
        mlflow.log_param("nested/nested/name", 5)
    finished_run = tracking.MlflowClient().get_run(run_id)
    # Validate params
    assert finished_run.data.params == {"name_1": "c", "name_2": "b", "nested/nested/name": "5"}


def test_log_params(tracking_uri_mock):
    expected_params = {"name_1": "c", "name_2": "b", "nested/nested/name": "5"}
    with start_run() as active_run:
        run_id = active_run.info.run_id
        mlflow.log_params(expected_params)
    finished_run = tracking.MlflowClient().get_run(run_id)
    # Validate params
    assert finished_run.data.params == {"name_1": "c", "name_2": "b", "nested/nested/name": "5"}


def test_log_batch_validates_entity_names_and_values(tracking_uri_mock):
    bad_kwargs = {
        "metrics": [
            [Metric(key="../bad/metric/name", value=0.3, timestamp=3, step=0)],
            [Metric(key="ok-name", value="non-numerical-value", timestamp=3, step=0)],
            [Metric(key="ok-name", value=0.3, timestamp="non-numerical-timestamp", step=0)],
        ],
        "params": [[Param(key="../bad/param/name", value="my-val")]],
        "tags": [[Param(key="../bad/tag/name", value="my-val")]],
    }
    with start_run() as active_run:
        for kwarg, bad_values in bad_kwargs.items():
            for bad_kwarg_value in bad_values:
                final_kwargs = {
                    "run_id":  active_run.info.run_id, "metrics": [], "params": [], "tags": [],
                }
                final_kwargs[kwarg] = bad_kwarg_value
                with pytest.raises(MlflowException) as e:
                    tracking.MlflowClient().log_batch(**final_kwargs)
                assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_log_artifact(tracking_uri_mock):
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
        expected_dir = os.path.join(run_artifact_dir, parent_dir) \
            if parent_dir is not None else run_artifact_dir
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
        expected_artifact_output_dir = os.path.join(run_artifact_dir, parent_dir) \
            if parent_dir is not None else run_artifact_dir
        dir_comparison = filecmp.dircmp(artifact_src_dir, expected_artifact_output_dir)
        assert len(dir_comparison.left_only) == 0
        assert len(dir_comparison.right_only) == 0
        assert len(dir_comparison.diff_files) == 0
        assert len(dir_comparison.funny_files) == 0


def test_uri_types():
    from mlflow.tracking import utils
    assert utils._is_local_uri("mlruns")
    assert utils._is_local_uri("./mlruns")
    assert utils._is_local_uri("file:///foo/mlruns")
    assert utils._is_local_uri("file:foo/mlruns")
    assert not utils._is_local_uri("https://whatever")
    assert not utils._is_local_uri("http://whatever")
    assert not utils._is_local_uri("databricks")
    assert not utils._is_local_uri("databricks:whatever")
    assert not utils._is_local_uri("databricks://whatever")

    assert utils._is_databricks_uri("databricks")
    assert utils._is_databricks_uri("databricks:whatever")
    assert utils._is_databricks_uri("databricks://whatever")
    assert not utils._is_databricks_uri("mlruns")
    assert not utils._is_databricks_uri("http://whatever")

    assert utils._is_http_uri("http://whatever")
    assert utils._is_http_uri("https://whatever")
    assert not utils._is_http_uri("file://whatever")
    assert not utils._is_http_uri("databricks://whatever")
    assert not utils._is_http_uri("mlruns")


def test_with_startrun():
    run_id = None
    import time
    t0 = int(time.time() * 1000)
    with mlflow.start_run() as active_run:
        assert mlflow.active_run() == active_run
        run_id = active_run.info.run_id
    t1 = int(time.time() * 1000)
    run_info = mlflow.tracking._get_store().get_run(run_id).info
    assert run_info.status == RunStatus.from_string("FINISHED")
    assert t0 <= run_info.end_time and run_info.end_time <= t1
    assert mlflow.active_run() is None


def test_parent_create_run(tracking_uri_mock):
    with mlflow.start_run() as parent_run:
        with pytest.raises(Exception, match='To start a nested run'):
            mlflow.start_run()
        with mlflow.start_run(nested=True) as child_run:
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
    with pytest.raises(MlflowException, matches='because it is in the deleted state.'):
        with mlflow.start_run(run_id=run_id):
            pass
    assert mlflow.active_run() is None


def test_start_run_exp_id_0(tracking_uri_mock, reset_active_experiment):
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
        assert mlflow.get_artifact_uri(artifact_path=artifact_path) == \
               tracking.artifact_utils.get_artifact_uri(
                run_id=active_run.info.run_id, artifact_path=artifact_path)


def test_search_runs(tracking_uri_mock, reset_active_experiment):
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


def test_search_runs_multiple_experiments(tracking_uri_mock, reset_active_experiment):
    experiment_ids = [mlflow.create_experiment("exp__{}".format(id)) for id in range(1, 4)]
    for eid in experiment_ids:
        with mlflow.start_run(experiment_id=eid):
            mlflow.log_metric("m0", 1)
            mlflow.log_metric("m_{}".format(eid), 2)

    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m0 > 0", ViewType.ALL)) == 3

    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m_1 > 0", ViewType.ALL)) == 1
    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m_2 = 2", ViewType.ALL)) == 1
    assert len(MlflowClient().search_runs(experiment_ids, "metrics.m_3 < 4", ViewType.ALL)) == 1
