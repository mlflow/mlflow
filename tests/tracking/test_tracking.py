import copy
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
from mlflow.entities import RunStatus, LifecycleStage, Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import start_run, end_run
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
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
    run = start_run()
    assert run.info.experiment_id == exp_id
    end_run()

    another_name = "another_experiment"
    mlflow.set_experiment(another_name)
    exp_id2 = mlflow.tracking.MlflowClient().get_experiment_by_name(another_name)
    another_run = start_run()
    assert another_run.info.experiment_id == exp_id2.experiment_id
    end_run()


def test_set_experiment_with_deleted_experiment_name(tracking_uri_mock):
    name = "dead_exp"
    mlflow.set_experiment(name)
    run = start_run()
    end_run()
    exp_id = run.info.experiment_id

    tracking.MlflowClient().delete_experiment(exp_id)

    with pytest.raises(MlflowException):
        mlflow.set_experiment(name)


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
    first_run = start_run()
    first_uuid = first_run.info.run_uuid
    with first_run:
        # Check that start_run() causes the run information to be persisted in the store
        persisted_run = tracking.MlflowClient().get_run(first_uuid)
        assert persisted_run is not None
        assert persisted_run.info == first_run.info
    finished_run = tracking.MlflowClient().get_run(first_uuid)
    assert finished_run.info.status == RunStatus.FINISHED
    # Launch a separate run that fails, verify the run status is FAILED and the run UUID is
    # different
    second_run = start_run()
    assert second_run.info.run_uuid != first_uuid
    with pytest.raises(Exception):
        with second_run:
            raise Exception("Failing run!")
    finished_run2 = tracking.MlflowClient().get_run(second_run.info.run_uuid)
    assert finished_run2.info.status == RunStatus.FAILED


def test_start_and_end_run(tracking_uri_mock):
    # Use the start_run() and end_run() APIs without a `with` block, verify they work.
    active_run = start_run()
    mlflow.log_metric("name_1", 25)
    end_run()
    finished_run = tracking.MlflowClient().get_run(active_run.info.run_uuid)
    # Validate metrics
    assert len(finished_run.data.metrics) == 1
    expected_pairs = {"name_1": 25}
    for metric in finished_run.data.metrics:
        assert expected_pairs[metric.key] == metric.value


def test_log_batch(tracking_uri_mock):
    expected_metrics = {"metric-key0": 1.0, "metric-key1": 4.0}
    expected_params = {"param-key0": "param-val0", "param-key1": "param-val1"}
    expected_tags = {"tag-key0": "tag-val0", "tag-key1": "tag-val1"}

    t = int(time.time())
    metrics = [Metric(key=key, value=value, timestamp=t) for key, value in expected_metrics.items()]
    params = [Param(key=key, value=value) for key, value in expected_params.items()]
    tags = [RunTag(key=key, value=value) for key, value in expected_tags.items()]

    active_run = start_run()
    run_uuid = active_run.info.run_uuid
    with active_run:
        mlflow.tracking.MlflowClient().log_batch(run_id=run_uuid, metrics=metrics, params=params,
                                                 tags=tags)
    finished_run = tracking.MlflowClient().get_run(run_uuid)
    # Validate metrics
    assert len(finished_run.data.metrics) == 2
    for metric in finished_run.data.metrics:
        assert expected_metrics[metric.key] == metric.value
    # Validate tags
    assert len(finished_run.data.tags) == 2
    for tag in finished_run.data.tags:
        assert expected_tags[tag.key] == tag.value
    # Validate params
    assert len(finished_run.data.params) == 2
    for param in finished_run.data.params:
        assert expected_params[param.key] == param.value


def test_log_metric(tracking_uri_mock):
    active_run = start_run()
    run_uuid = active_run.info.run_uuid
    with active_run:
        mlflow.log_metric("name_1", 25)
        mlflow.log_metric("name_2", -3)
        mlflow.log_metric("name_1", 30)
        mlflow.log_metric("nested/nested/name", 40)
    finished_run = tracking.MlflowClient().get_run(run_uuid)
    # Validate metrics
    assert len(finished_run.data.metrics) == 3
    expected_pairs = {"name_1": 30, "name_2": -3, "nested/nested/name": 40}
    for metric in finished_run.data.metrics:
        assert expected_pairs[metric.key] == metric.value


def test_log_metrics(tracking_uri_mock):
    active_run = start_run()
    run_uuid = active_run.info.run_uuid
    expected_metrics = {"name_1": 30, "name_2": -3, "nested/nested/name": 40}
    with active_run:
        mlflow.log_metric("name_1", 25)
        mlflow.log_metric("nested/nested/name", 45)
        mlflow.log_metrics(expected_metrics)
    finished_run = tracking.MlflowClient().get_run(run_uuid)
    # Validate metrics
    assert len(finished_run.data.metrics) == 3
    for metric in finished_run.data.metrics:
        assert expected_metrics[metric.key] == metric.value


@pytest.fixture
def get_store_mock(tmpdir):
    with mock.patch("mlflow.store.file_store.FileStore.log_batch") as _get_store_mock:
        yield _get_store_mock


def test_log_batch_validations(tracking_uri_mock, get_store_mock):
    # Log same param with different values, verify error before we hit the store
    # Log metric/param/tag with bad keys, verify error before we hit the store
    # TODO: Do we want to validate on long keys in the client? Would be a breaking behavior change
    metrics_with_bad_key = [Metric("good-metric-key", 1.0, 0), Metric("super-long-bad-key" * 1000, 4.0, 0)]
    params_with_bad_key = [Param("good-param-key", "hi"), Param("super-long-bad-key" * 1000, "but-good-val")]
    params_with_bad_val = [Param("good-param-key", "hi"), Param("another-good-key", "but-bad-val" * 1000)]
    tags_with_bad_key = [RunTag("good-tag-key", "hi"), RunTag("super-long-bad-key" * 1000, "but-good-val")]
    tags_with_bad_val = [RunTag("good-tag-key", "hi"), RunTag("another-good-key", "but-bad-val" * 1000)]

    too_many_metrics = [Metric("metric-key-%s" % i, 1, 0) for i in range(1001)]
    too_many_params = [Param("param-key-%s" % i, "b") for i in range(101)]
    too_many_tags = [RunTag("tag-key-%s" % i, "b") for i in range(101)]

    overwriting_param = [Param("key", "val"), Param("key", "different-val")]

    good_kwargs = {"metrics": [], "params": [], "tags": []}
    bad_kwargs = {
        "metrics": [metrics_with_bad_key, too_many_metrics],
        "params": [params_with_bad_key, params_with_bad_val, too_many_params, overwriting_param],
        "tags": [tags_with_bad_key, tags_with_bad_val, too_many_tags],
    }
    active_run = start_run()
    run_uuid = active_run.info.run_uuid
    with active_run:
        for arg_name, arg_values in bad_kwargs.items():
            for arg_value in arg_values:
                final_kwargs = copy.deepcopy(good_kwargs)
                final_kwargs[arg_name] = arg_value
                with pytest.raises(MlflowException):
                    mlflow.tracking.MlflowClient().log_batch(run_id=run_uuid, **final_kwargs)
                assert get_store_mock.not_called()
        # Test the case where there are too many entities in aggregate
        too_many_entities = {
            "metrics": too_many_metrics[:900],
            "tags": too_many_tags[:50],
            "params": too_many_params[:51],
        }
        with pytest.raises(MlflowException):
            mlflow.tracking.MlflowClient().log_batch(run_id=run_uuid, **too_many_entities)
            assert get_store_mock.not_called()


def test_set_tags(tracking_uri_mock):
    expected_tags = {"name_1": "c", "name_2": "b", "nested/nested/name": "5"}
    active_run = start_run()
    run_uuid = active_run.info.run_uuid
    with active_run:
        mlflow.set_tags(expected_tags)
    finished_run = tracking.MlflowClient().get_run(run_uuid)
    # Validate tags
    assert len(finished_run.data.tags) == 3
    for tag in finished_run.data.tags:
        assert expected_tags[tag.key] == tag.value


def test_log_metric_validation(tracking_uri_mock):
    active_run = start_run()
    run_uuid = active_run.info.run_uuid
    with active_run:
        mlflow.log_metric("name_1", "apple")
    finished_run = tracking.MlflowClient().get_run(run_uuid)
    assert len(finished_run.data.metrics) == 0


def test_log_param(tracking_uri_mock):
    active_run = start_run()
    run_uuid = active_run.info.run_uuid
    with active_run:
        mlflow.log_param("name_1", "a")
        mlflow.log_param("name_2", "b")
        mlflow.log_param("name_1", "c")
        mlflow.log_param("nested/nested/name", 5)
    finished_run = tracking.MlflowClient().get_run(run_uuid)
    # Validate params
    assert len(finished_run.data.params) == 3
    expected_pairs = {"name_1": "c", "name_2": "b", "nested/nested/name": "5"}
    for param in finished_run.data.params:
        assert expected_pairs[param.key] == param.value


def test_log_params(tracking_uri_mock):
    expected_params = {"name_1": "c", "name_2": "b", "nested/nested/name": "5"}
    active_run = start_run()
    run_uuid = active_run.info.run_uuid
    with active_run:
        mlflow.log_params(expected_params)
    finished_run = tracking.MlflowClient().get_run(run_uuid)
    # Validate params
    assert len(finished_run.data.params) == 3
    for param in finished_run.data.params:
        assert expected_params[param.key] == param.value


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
            run_artifact_dir = mlflow.get_artifact_uri()
            mlflow.log_artifact(path0, parent_dir)
        expected_dir = os.path.join(run_artifact_dir, parent_dir) \
            if parent_dir is not None else run_artifact_dir
        assert os.listdir(expected_dir) == [os.path.basename(path0)]
        logged_artifact_path = os.path.join(expected_dir, path0)
        assert filecmp.cmp(logged_artifact_path, path0, shallow=False)
    # Log multiple artifacts, verify they exist in the directory returned by get_artifact_uri
    for parent_dir in artifact_parent_dirs:
        with start_run():
            run_artifact_dir = mlflow.get_artifact_uri()
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
        run_id = active_run.info.run_uuid
    t1 = int(time.time() * 1000)
    run_info = mlflow.tracking._get_store().get_run(run_id).info
    assert run_info.status == RunStatus.from_string("FINISHED")
    assert t0 <= run_info.end_time and run_info.end_time <= t1
    assert mlflow.active_run() is None


def test_parent_create_run(tracking_uri_mock):
    parent_run = mlflow.start_run()
    with pytest.raises(Exception, match='To start a nested run'):
        mlflow.start_run()
    child_run = mlflow.start_run(nested=True)
    grand_child_run = mlflow.start_run(nested=True)

    def verify_has_parent_id_tag(child_id, expected_parent_id):
        tags = tracking.MlflowClient().get_run(child_id).data.tags
        assert any([t.key == MLFLOW_PARENT_RUN_ID and t.value == expected_parent_id for t in tags])

    verify_has_parent_id_tag(child_run.info.run_uuid, parent_run.info.run_uuid)
    verify_has_parent_id_tag(grand_child_run.info.run_uuid, child_run.info.run_uuid)

    mlflow.end_run()
    mlflow.end_run()
    mlflow.end_run()
    assert mlflow.active_run() is None


def test_start_deleted_run():
    run_id = None
    with mlflow.start_run() as active_run:
        run_id = active_run.info.run_uuid
    tracking.MlflowClient().delete_run(run_id)
    with pytest.raises(MlflowException, matches='because it is in the deleted state.'):
        with mlflow.start_run(run_uuid=run_id):
            pass
    assert mlflow.active_run() is None


def test_start_run_exp_id_0(tracking_uri_mock, reset_active_experiment):
    mlflow.set_experiment("some-experiment")
    # Create a run and verify that the current active experiment is the one we just set
    with mlflow.start_run() as active_run:
        exp_id = active_run.info.experiment_id
        assert exp_id != 0
        assert MlflowClient().get_experiment(exp_id).name == "some-experiment"
    # Set experiment ID to 0 when creating a run, verify that the specified experiment ID is honored
    with mlflow.start_run(experiment_id=0) as active_run:
        assert active_run.info.experiment_id == 0


def test_get_artifact_uri_with_artifact_path_unspecified_returns_artifact_root_dir():
    with mlflow.start_run() as active_run:
        assert mlflow.get_artifact_uri(artifact_path=None) == active_run.info.artifact_uri


def test_get_artifact_uri_uses_currently_active_run_id():
    artifact_path = "artifact"
    with mlflow.start_run() as active_run:
        assert mlflow.get_artifact_uri(artifact_path=artifact_path) ==\
            tracking.utils.get_artifact_uri(
                run_id=active_run.info.run_uuid, artifact_path=artifact_path)
