import random
from contextlib import contextmanager
import filecmp
import os
import tempfile
import shutil

import mock
import pytest

from mlflow.store.file_store import FileStore
from mlflow.entities.run_status import RunStatus
from mlflow import tracking
import mlflow


@contextmanager
def temp_directory():
    name = tempfile.mkdtemp()
    try:
        yield name
    finally:
        shutil.rmtree(name)


def test_create_experiment():
    with pytest.raises(TypeError):
        tracking.create_experiment()

    with pytest.raises(Exception):
        tracking.create_experiment(None)

    with pytest.raises(Exception):
        tracking.create_experiment("")

    with temp_directory() as tmp_dir, mock.patch("mlflow.tracking._get_store") as get_store_mock:
        get_store_mock.return_value = FileStore(tmp_dir)
        exp_id = tracking.create_experiment("Some random experiment name %d" % random.randint(1, 1e6))
        assert exp_id is not None


def test_start_run_context_manager():
    with temp_directory() as tmp_dir, mock.patch("mlflow.tracking._get_store") as get_store_mock:
        get_store_mock.return_value = FileStore(tmp_dir)
        first_run = tracking.start_run()
        store = first_run.store
        first_uuid = first_run.run_info.run_uuid
        with first_run:
            # Check that start_run() causes the run information to be persisted in the store
            persisted_run = store.get_run(first_uuid)
            assert persisted_run is not None
            assert persisted_run.info == first_run.run_info
        finished_run = store.get_run(first_uuid)
        assert finished_run.info.status == RunStatus.FINISHED
        # Launch a separate run that fails, verify the run status is FAILED and the run UUID is
        # different
        second_run = tracking.start_run()
        assert second_run.run_info.run_uuid != first_uuid
        with pytest.raises(Exception):
            with second_run:
                raise Exception("Failing run!")
        finished_run2 = store.get_run(second_run.run_info.run_uuid)
        assert finished_run2.info == second_run.run_info
        assert finished_run2.info.status == RunStatus.FAILED
        # Nested runs return original run
        with tracking.start_run():
            active_run = tracking._active_run
            new_run = tracking.start_run()
            assert active_run == new_run


def test_start_and_end_run():
    with temp_directory() as tmp_dir, mock.patch("mlflow.tracking._get_store") as get_store_mock:
        get_store_mock.return_value = FileStore(tmp_dir)
        # Use the start_run() and end_run() APIs without a `with` block, verify they work.
        active_run = tracking.start_run()
        mlflow.log_metric("name_1", 25)
        tracking.end_run()
        finished_run = active_run.store.get_run(active_run.run_info.run_uuid)
        # Validate metrics
        assert len(finished_run.data.metrics) == 1
        expected_pairs = {"name_1": 25}
        for metric in finished_run.data.metrics:
            assert expected_pairs[metric.key] == metric.value


def test_log_metric():
    with temp_directory() as tmp_dir, mock.patch("mlflow.tracking._get_store") as get_store_mock:
        get_store_mock.return_value = FileStore(tmp_dir)
        active_run = tracking.start_run()
        run_uuid = active_run.run_info.run_uuid
        with active_run:
            mlflow.log_metric("name_1", 25)
            mlflow.log_metric("name_2", -3)
            mlflow.log_metric("name_1", 30)
        finished_run = active_run.store.get_run(run_uuid)
        # Validate metrics
        assert len(finished_run.data.metrics) == 2
        expected_pairs = {"name_1": 30, "name_2": -3}
        for metric in finished_run.data.metrics:
            assert expected_pairs[metric.key] == metric.value


def test_log_metric_validation():
    with temp_directory() as tmp_dir, mock.patch("mlflow.tracking._get_store") as get_store_mock:
        get_store_mock.return_value = FileStore(tmp_dir)
        active_run = tracking.start_run()
        run_uuid = active_run.run_info.run_uuid
        with active_run:
            mlflow.log_metric("name_1", "apple")
        finished_run = active_run.store.get_run(run_uuid)
        assert len(finished_run.data.metrics) == 0


def test_log_param():
    with temp_directory() as tmp_dir, mock.patch("mlflow.tracking._get_store") as get_store_mock:
        get_store_mock.return_value = FileStore(tmp_dir)
        active_run = tracking.start_run()
        run_uuid = active_run.run_info.run_uuid
        with active_run:
            mlflow.log_param("name_1", "a")
            mlflow.log_param("name_2", "b")
            mlflow.log_param("name_1", "c")
        finished_run = active_run.store.get_run(run_uuid)
        # Validate params
        assert len(finished_run.data.params) == 2
        expected_pairs = {"name_1": "c", "name_2": "b"}
        for param in finished_run.data.params:
            assert expected_pairs[param.key] == param.value


def test_log_artifact():
    with temp_directory() as tmp_dir, temp_directory() as artifact_src_dir, \
            mock.patch("mlflow.tracking._get_store") as get_store_mock:
        get_store_mock.return_value = FileStore(tmp_dir)
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
            with tracking.start_run():
                run_artifact_dir = mlflow.get_artifact_uri()
                mlflow.log_artifact(path0, parent_dir)
            expected_dir = os.path.join(run_artifact_dir, parent_dir) \
                if parent_dir is not None else run_artifact_dir
            assert os.listdir(expected_dir) == [os.path.basename(path0)]
            logged_artifact_path = os.path.join(expected_dir, path0)
            assert filecmp.cmp(logged_artifact_path, path0, shallow=False)
        # Log multiple artifacts, verify they exist in the directory returned by get_artifact_uri
        for parent_dir in artifact_parent_dirs:
            with tracking.start_run():
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
