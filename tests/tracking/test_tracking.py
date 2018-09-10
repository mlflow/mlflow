import random
from contextlib import contextmanager
import filecmp
import os
import tempfile
import shutil

import mock
import pytest

from mlflow.store.file_store import FileStore
from mlflow.entities import RunStatus
from mlflow import tracking
from mlflow.tracking.fluent import start_run, end_run
import mlflow


def test_create_experiment():
    with pytest.raises(TypeError):
        mlflow.create_experiment()

    with pytest.raises(Exception):
        mlflow.create_experiment(None)

    with pytest.raises(Exception):
        mlflow.create_experiment("")

    try:
        tracking.set_tracking_uri(tempfile.mkdtemp())
        exp_id = mlflow.create_experiment(
            "Some random experiment name %d" % random.randint(1, 1e6))
        assert exp_id is not None
    finally:
        tracking.set_tracking_uri(None)


def test_no_nested_run():
    try:
        tracking.set_tracking_uri(tempfile.mkdtemp())
        first_run = start_run()
        with first_run:
            with pytest.raises(Exception):
                start_run()
    finally:
        tracking.set_tracking_uri(None)


def test_start_run_context_manager():
    try:
        tracking.set_tracking_uri(tempfile.mkdtemp())
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
    finally:
        tracking.set_tracking_uri(None)


def test_start_and_end_run():
    try:
        tracking.set_tracking_uri(tempfile.mkdtemp())
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
    finally:
        tracking.set_tracking_uri(None)


def test_log_metric():
    try:
        tracking.set_tracking_uri(tempfile.mkdtemp())
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
    finally:
        tracking.set_tracking_uri(None)


def test_log_metric_validation():
    try:
        tracking.set_tracking_uri(tempfile.mkdtemp())
        active_run = start_run()
        run_uuid = active_run.info.run_uuid
        with active_run:
            mlflow.log_metric("name_1", "apple")
        finished_run = tracking.MlflowClient().get_run(run_uuid)
        assert len(finished_run.data.metrics) == 0
    finally:
        tracking.set_tracking_uri(None)


def test_log_param():
    try:
        tracking.set_tracking_uri(tempfile.mkdtemp())
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
    finally:
        tracking.set_tracking_uri(None)


def test_log_artifact():
    try:
        tracking.set_tracking_uri(tempfile.mkdtemp())
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
    finally:
        tracking.set_tracking_uri(None)


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
    runId = None
    import time
    t0 = int(time.time() * 1000)
    with mlflow.start_run() as active_run:
        assert mlflow.active_run() == active_run
        runId = active_run.info.run_uuid
    t1 = int(time.time() * 1000)
    run_info = mlflow.tracking._get_store().get_run(runId).info
    assert run_info.status == RunStatus.from_string("FINISHED")
    assert t0 <= run_info.end_time and run_info.end_time <= t1
    assert mlflow.active_run() is None
