#!/usr/bin/env python
import os
import posixpath
import random
import shutil
import tempfile
import time
import unittest
import uuid

import pytest
from unittest import mock

from mlflow.entities import (
    Metric,
    Param,
    RunTag,
    ViewType,
    LifecycleStage,
    RunStatus,
    RunData,
    ExperimentTag,
)
from mlflow.exceptions import MlflowException, MissingConfigException
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.file_store import FileStore
from mlflow.utils.file_utils import write_yaml, read_yaml, path_to_local_file_uri, TempDir
from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    RESOURCE_DOES_NOT_EXIST,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
)

from tests.helper_functions import random_int, random_str, safe_edit_yaml
from tests.store.tracking import AbstractStoreTest

FILESTORE_PACKAGE = "mlflow.store.tracking.file_store"


class TestFileStore(unittest.TestCase, AbstractStoreTest):
    ROOT_LOCATION = tempfile.gettempdir()

    def create_test_run(self):
        fs = FileStore(self.test_root)
        return self._create_run(fs)

    def setUp(self):
        self._create_root(TestFileStore.ROOT_LOCATION)
        self.maxDiff = None

    def get_store(self):
        return FileStore(self.test_root)

    def _create_root(self, root):
        self.test_root = os.path.join(root, "test_file_store_%d" % random_int())
        os.mkdir(self.test_root)
        self.experiments = [str(random_int(100, int(1e9))) for _ in range(3)]
        self.exp_data = {}
        self.run_data = {}
        # Include default experiment
        self.experiments.append(FileStore.DEFAULT_EXPERIMENT_ID)
        for exp in self.experiments:
            # create experiment
            exp_folder = os.path.join(self.test_root, str(exp))
            os.makedirs(exp_folder)
            d = {"experiment_id": exp, "name": random_str(), "artifact_location": exp_folder}
            self.exp_data[exp] = d
            write_yaml(exp_folder, FileStore.META_DATA_FILE_NAME, d)
            # add runs
            self.exp_data[exp]["runs"] = []
            for _ in range(2):
                run_id = uuid.uuid4().hex
                self.exp_data[exp]["runs"].append(run_id)
                run_folder = os.path.join(exp_folder, run_id)
                os.makedirs(run_folder)
                run_info = {
                    "run_uuid": run_id,
                    "run_id": run_id,
                    "experiment_id": exp,
                    "user_id": random_str(random_int(10, 25)),
                    "status": random.choice(RunStatus.all_status()),
                    "start_time": random_int(1, 10),
                    "end_time": random_int(20, 30),
                    "tags": [],
                    "artifact_uri": os.path.join(run_folder, FileStore.ARTIFACTS_FOLDER_NAME),
                }
                write_yaml(run_folder, FileStore.META_DATA_FILE_NAME, run_info)
                self.run_data[run_id] = run_info
                # tags
                os.makedirs(os.path.join(run_folder, FileStore.TAGS_FOLDER_NAME))
                # params
                params_folder = os.path.join(run_folder, FileStore.PARAMS_FOLDER_NAME)
                os.makedirs(params_folder)
                params = {}
                for _ in range(5):
                    param_name = random_str(random_int(10, 12))
                    param_value = random_str(random_int(10, 15))
                    param_file = os.path.join(params_folder, param_name)
                    with open(param_file, "w") as f:
                        f.write(param_value)
                    params[param_name] = param_value
                self.run_data[run_id]["params"] = params
                # metrics
                metrics_folder = os.path.join(run_folder, FileStore.METRICS_FOLDER_NAME)
                os.makedirs(metrics_folder)
                metrics = {}
                for _ in range(3):
                    metric_name = random_str(random_int(10, 12))
                    timestamp = int(time.time())
                    metric_file = os.path.join(metrics_folder, metric_name)
                    values = []
                    for _ in range(10):
                        metric_value = random_int(100, 2000)
                        timestamp += random_int(10000, 2000000)
                        values.append((timestamp, metric_value))
                        with open(metric_file, "a") as f:
                            f.write("%d %d\n" % (timestamp, metric_value))
                    metrics[metric_name] = values
                self.run_data[run_id]["metrics"] = metrics
                # artifacts
                os.makedirs(os.path.join(run_folder, FileStore.ARTIFACTS_FOLDER_NAME))

    def tearDown(self):
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_valid_root(self):
        # Test with valid root
        file_store = FileStore(self.test_root)
        try:
            file_store._check_root_dir()
        except Exception as e:
            self.fail("test_valid_root raised exception '%s'" % e.message)

        # Test removing root
        second_file_store = FileStore(self.test_root)
        shutil.rmtree(self.test_root)
        with self.assertRaises(Exception):
            second_file_store._check_root_dir()

    def test_list_experiments(self):
        fs = FileStore(self.test_root)
        for exp in fs.list_experiments():
            exp_id = exp.experiment_id
            self.assertTrue(exp_id in self.experiments)
            self.assertEqual(exp.name, self.exp_data[exp_id]["name"])
            self.assertEqual(exp.artifact_location, self.exp_data[exp_id]["artifact_location"])

    def test_list_experiments_paginated(self):
        fs = FileStore(self.test_root)
        for _ in range(10):
            fs.create_experiment(random_str(12))
        exps1 = fs.list_experiments(max_results=4, page_token=None)
        self.assertEqual(len(exps1), 4)
        self.assertIsNotNone(exps1.token)
        exps2 = fs.list_experiments(max_results=4, page_token=None)
        self.assertEqual(len(exps2), 4)
        self.assertIsNotNone(exps2.token)
        self.assertNotEqual(exps1, exps2)
        exps3 = fs.list_experiments(max_results=500, page_token=exps2.token)
        self.assertLessEqual(len(exps3), 500)
        if len(exps3) < 500:
            self.assertIsNone(exps3.token)

    def _verify_experiment(self, fs, exp_id):
        exp = fs.get_experiment(exp_id)
        self.assertEqual(exp.experiment_id, exp_id)
        self.assertEqual(exp.name, self.exp_data[exp_id]["name"])
        self.assertEqual(exp.artifact_location, self.exp_data[exp_id]["artifact_location"])

    def test_get_experiment(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            self._verify_experiment(fs, exp_id)

        # test that fake experiments dont exist.
        # look for random experiment ids between 8000, 15000 since created ones are (100, 2000)
        for exp_id in set(random_int(8000, 15000) for x in range(20)):
            with self.assertRaises(Exception):
                fs.get_experiment(exp_id)

    def test_get_experiment_int_experiment_id_backcompat(self):
        fs = FileStore(self.test_root)
        exp_id = FileStore.DEFAULT_EXPERIMENT_ID
        root_dir = os.path.join(self.test_root, exp_id)
        with safe_edit_yaml(root_dir, "meta.yaml", self._experiment_id_edit_func):
            self._verify_experiment(fs, exp_id)

    def test_get_experiment_by_name(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            name = self.exp_data[exp_id]["name"]
            exp = fs.get_experiment_by_name(name)
            self.assertEqual(exp.experiment_id, exp_id)
            self.assertEqual(exp.name, self.exp_data[exp_id]["name"])
            self.assertEqual(exp.artifact_location, self.exp_data[exp_id]["artifact_location"])

        # test that fake experiments dont exist.
        # look up experiments with names of length 15 since created ones are of length 10
        for exp_names in set(random_str(15) for x in range(20)):
            exp = fs.get_experiment_by_name(exp_names)
            self.assertIsNone(exp)

    def test_create_first_experiment(self):
        fs = FileStore(self.test_root)
        fs.list_experiments = mock.Mock(return_value=[])
        fs._create_experiment_with_id = mock.Mock()
        fs.create_experiment(random_str())
        fs._create_experiment_with_id.assert_called_once()
        experiment_id = fs._create_experiment_with_id.call_args[0][1]
        self.assertEqual(experiment_id, FileStore.DEFAULT_EXPERIMENT_ID)

    def test_create_experiment(self):
        fs = FileStore(self.test_root)

        # Error cases
        with self.assertRaises(Exception):
            fs.create_experiment(None)
        with self.assertRaises(Exception):
            fs.create_experiment("")

        exp_id_ints = (int(exp_id) for exp_id in self.experiments)
        next_id = str(max(exp_id_ints) + 1)
        name = random_str(25)  # since existing experiments are 10 chars long
        created_id = fs.create_experiment(name)
        # test that newly created experiment matches expected id
        self.assertEqual(created_id, next_id)

        # get the new experiment (by id) and verify (by name)
        exp1 = fs.get_experiment(created_id)
        self.assertEqual(exp1.name, name)
        self.assertEqual(
            exp1.artifact_location,
            path_to_local_file_uri(posixpath.join(self.test_root, created_id)),
        )

        # get the new experiment (by name) and verify (by id)
        exp2 = fs.get_experiment_by_name(name)
        self.assertEqual(exp2.experiment_id, created_id)

    def test_create_experiment_appends_to_artifact_uri_path_correctly(self):
        cases = [
            ("path/to/local/folder", "path/to/local/folder/{e}"),
            ("/path/to/local/folder", "/path/to/local/folder/{e}"),
            ("#path/to/local/folder?", "#path/to/local/folder?/{e}"),
            ("file:path/to/local/folder", "file:path/to/local/folder/{e}"),
            ("file:///path/to/local/folder", "file:///path/to/local/folder/{e}"),
            ("file:path/to/local/folder?param=value", "file:path/to/local/folder/{e}?param=value"),
            ("file:///path/to/local/folder", "file:///path/to/local/folder/{e}"),
            (
                "file:///path/to/local/folder?param=value#fragment",
                "file:///path/to/local/folder/{e}?param=value#fragment",
            ),
            ("s3://bucket/path/to/root", "s3://bucket/path/to/root/{e}"),
            (
                "s3://bucket/path/to/root?creds=mycreds",
                "s3://bucket/path/to/root/{e}?creds=mycreds",
            ),
            (
                "dbscheme+driver://root@host/dbname?creds=mycreds#myfragment",
                "dbscheme+driver://root@host/dbname/{e}?creds=mycreds#myfragment",
            ),
            (
                "dbscheme+driver://root:password@hostname.com?creds=mycreds#myfragment",
                "dbscheme+driver://root:password@hostname.com/{e}?creds=mycreds#myfragment",
            ),
            (
                "dbscheme+driver://root:password@hostname.com/mydb?creds=mycreds#myfragment",
                "dbscheme+driver://root:password@hostname.com/mydb/{e}?creds=mycreds#myfragment",
            ),
        ]

        for artifact_root_uri, expected_artifact_uri_format in cases:
            with TempDir() as tmp:
                fs = FileStore(tmp.path(), artifact_root_uri)
                exp_id = fs.create_experiment("exp")
                exp = fs.get_experiment(exp_id)
                self.assertEqual(
                    exp.artifact_location, expected_artifact_uri_format.format(e=exp_id)
                )

    def test_create_experiment_with_tags_works_correctly(self):
        fs = FileStore(self.test_root)

        created_id = fs.create_experiment(
            "heresAnExperiment",
            "heresAnArtifact",
            [ExperimentTag("key1", "val1"), ExperimentTag("key2", "val2")],
        )
        experiment = fs.get_experiment(created_id)
        assert len(experiment.tags) == 2
        assert experiment.tags["key1"] == "val1"
        assert experiment.tags["key2"] == "val2"

    def test_create_duplicate_experiments(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            name = self.exp_data[exp_id]["name"]
            with self.assertRaises(Exception):
                fs.create_experiment(name)

    def _extract_ids(self, experiments):
        return [e.experiment_id for e in experiments]

    def test_delete_restore_experiment(self):
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        exp_name = self.exp_data[exp_id]["name"]

        # delete it
        fs.delete_experiment(exp_id)
        self.assertTrue(exp_id not in self._extract_ids(fs.list_experiments(ViewType.ACTIVE_ONLY)))
        self.assertTrue(exp_id in self._extract_ids(fs.list_experiments(ViewType.DELETED_ONLY)))
        self.assertTrue(exp_id in self._extract_ids(fs.list_experiments(ViewType.ALL)))
        self.assertEqual(fs.get_experiment(exp_id).lifecycle_stage, LifecycleStage.DELETED)

        # restore it
        fs.restore_experiment(exp_id)
        restored_1 = fs.get_experiment(exp_id)
        self.assertEqual(restored_1.experiment_id, exp_id)
        self.assertEqual(restored_1.name, exp_name)
        restored_2 = fs.get_experiment_by_name(exp_name)
        self.assertEqual(restored_2.experiment_id, exp_id)
        self.assertEqual(restored_2.name, exp_name)
        self.assertTrue(exp_id in self._extract_ids(fs.list_experiments(ViewType.ACTIVE_ONLY)))
        self.assertTrue(exp_id not in self._extract_ids(fs.list_experiments(ViewType.DELETED_ONLY)))
        self.assertTrue(exp_id in self._extract_ids(fs.list_experiments(ViewType.ALL)))
        self.assertEqual(fs.get_experiment(exp_id).lifecycle_stage, LifecycleStage.ACTIVE)

    def test_rename_experiment(self):
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]

        # Error cases
        with self.assertRaises(Exception):
            fs.rename_experiment(exp_id, None)
        with self.assertRaises(Exception):
            # test that names of existing experiments are checked before renaming
            other_exp_id = None
            for exp in self.experiments:
                if exp != exp_id:
                    other_exp_id = exp
                    break
            fs.rename_experiment(exp_id, fs.get_experiment(other_exp_id).name)

        exp_name = self.exp_data[exp_id]["name"]
        new_name = exp_name + "!!!"
        self.assertNotEqual(exp_name, new_name)
        self.assertEqual(fs.get_experiment(exp_id).name, exp_name)
        fs.rename_experiment(exp_id, new_name)
        self.assertEqual(fs.get_experiment(exp_id).name, new_name)

        # Ensure that we cannot rename deleted experiments.
        fs.delete_experiment(exp_id)
        with pytest.raises(
            Exception, match="Cannot rename experiment in non-active lifecycle stage"
        ) as e:
            fs.rename_experiment(exp_id, exp_name)
        assert "non-active lifecycle" in str(e.value)
        self.assertEqual(fs.get_experiment(exp_id).name, new_name)

        # Restore the experiment, and confirm that we acn now rename it.
        fs.restore_experiment(exp_id)
        self.assertEqual(fs.get_experiment(exp_id).name, new_name)
        fs.rename_experiment(exp_id, exp_name)
        self.assertEqual(fs.get_experiment(exp_id).name, exp_name)

    def test_delete_restore_run(self):
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]["runs"][0]
        # Should not throw.
        assert fs.get_run(run_id).info.lifecycle_stage == "active"
        fs.delete_run(run_id)
        assert fs.get_run(run_id).info.lifecycle_stage == "deleted"
        fs.restore_run(run_id)
        assert fs.get_run(run_id).info.lifecycle_stage == "active"

    def test_hard_delete_run(self):
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]["runs"][0]
        fs._hard_delete_run(run_id)
        with self.assertRaises(MlflowException):
            fs.get_run(run_id)
        with self.assertRaises(MlflowException):
            fs.get_all_tags(run_id)
        with self.assertRaises(MlflowException):
            fs.get_all_metrics(run_id)
        with self.assertRaises(MlflowException):
            fs.get_all_params(run_id)

    def test_get_deleted_runs(self):
        fs = FileStore(self.test_root)
        exp_id = self.experiments[0]
        run_id = self.exp_data[exp_id]["runs"][0]
        fs.delete_run(run_id)
        deleted_runs = fs._get_deleted_runs()
        assert len(deleted_runs) == 1
        assert deleted_runs[0] == run_id

    def test_create_run_appends_to_artifact_uri_path_correctly(self):
        cases = [
            ("path/to/local/folder", "path/to/local/folder/{e}/{r}/artifacts"),
            ("/path/to/local/folder", "/path/to/local/folder/{e}/{r}/artifacts"),
            ("#path/to/local/folder?", "#path/to/local/folder?/{e}/{r}/artifacts"),
            ("file:path/to/local/folder", "file:path/to/local/folder/{e}/{r}/artifacts"),
            ("file:///path/to/local/folder", "file:///path/to/local/folder/{e}/{r}/artifacts"),
            (
                "file:path/to/local/folder?param=value",
                "file:path/to/local/folder/{e}/{r}/artifacts?param=value",
            ),
            ("file:///path/to/local/folder", "file:///path/to/local/folder/{e}/{r}/artifacts"),
            (
                "file:///path/to/local/folder?param=value#fragment",
                "file:///path/to/local/folder/{e}/{r}/artifacts?param=value#fragment",
            ),
            ("s3://bucket/path/to/root", "s3://bucket/path/to/root/{e}/{r}/artifacts"),
            (
                "s3://bucket/path/to/root?creds=mycreds",
                "s3://bucket/path/to/root/{e}/{r}/artifacts?creds=mycreds",
            ),
            (
                "dbscheme+driver://root@host/dbname?creds=mycreds#myfragment",
                "dbscheme+driver://root@host/dbname/{e}/{r}/artifacts?creds=mycreds#myfragment",
            ),
            (
                "dbscheme+driver://root:password@hostname.com?creds=mycreds#myfragment",
                "dbscheme+driver://root:password@hostname.com/{e}/{r}/artifacts"
                "?creds=mycreds#myfragment",
            ),
            (
                "dbscheme+driver://root:password@hostname.com/mydb?creds=mycreds#myfragment",
                "dbscheme+driver://root:password@hostname.com/mydb/{e}/{r}/artifacts"
                "?creds=mycreds#myfragment",
            ),
        ]

        for artifact_root_uri, expected_artifact_uri_format in cases:
            with TempDir() as tmp:
                fs = FileStore(tmp.path(), artifact_root_uri)
                exp_id = fs.create_experiment("exp")
                run = fs.create_run(experiment_id=exp_id, user_id="user", start_time=0, tags=[])
                self.assertEqual(
                    run.info.artifact_uri,
                    expected_artifact_uri_format.format(e=exp_id, r=run.info.run_id),
                )

    def test_create_run_in_deleted_experiment(self):
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        # delete it
        fs.delete_experiment(exp_id)
        with pytest.raises(Exception, match="Could not create run under non-active experiment"):
            fs.create_run(exp_id, "user", 0, [])

    def test_create_run_returns_expected_run_data(self):
        fs = FileStore(self.test_root)
        no_tags_run = fs.create_run(
            experiment_id=FileStore.DEFAULT_EXPERIMENT_ID, user_id="user", start_time=0, tags=[]
        )
        assert isinstance(no_tags_run.data, RunData)
        assert len(no_tags_run.data.tags) == 0

        tags_dict = {
            "my_first_tag": "first",
            "my-second-tag": "2nd",
        }
        tags_entities = [RunTag(key, value) for key, value in tags_dict.items()]
        tags_run = fs.create_run(
            experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
            user_id="user",
            start_time=0,
            tags=tags_entities,
        )
        assert isinstance(tags_run.data, RunData)
        assert tags_run.data.tags == tags_dict

    def _experiment_id_edit_func(self, old_dict):
        old_dict["experiment_id"] = int(old_dict["experiment_id"])
        return old_dict

    def _verify_run(self, fs, run_id):
        run = fs.get_run(run_id)
        run_info = self.run_data[run_id]
        run_info.pop("metrics", None)
        run_info.pop("params", None)
        run_info.pop("tags", None)
        run_info["lifecycle_stage"] = LifecycleStage.ACTIVE
        run_info["status"] = RunStatus.to_string(run_info["status"])
        self.assertEqual(run_info, dict(run.info))

    def test_get_run(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_id in runs:
                self._verify_run(fs, run_id)

    def test_get_run_int_experiment_id_backcompat(self):
        fs = FileStore(self.test_root)
        exp_id = FileStore.DEFAULT_EXPERIMENT_ID
        run_id = self.exp_data[exp_id]["runs"][0]
        root_dir = os.path.join(self.test_root, exp_id, run_id)
        with safe_edit_yaml(root_dir, "meta.yaml", self._experiment_id_edit_func):
            self._verify_run(fs, run_id)

    def test_list_run_infos(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            run_infos = fs.list_run_infos(exp_id, run_view_type=ViewType.ALL)
            for run_info in run_infos:
                run_id = run_info.run_id
                dict_run_info = self.run_data[run_id]
                dict_run_info.pop("metrics")
                dict_run_info.pop("params")
                dict_run_info.pop("tags")
                dict_run_info["lifecycle_stage"] = LifecycleStage.ACTIVE
                dict_run_info["status"] = RunStatus.to_string(dict_run_info["status"])
                self.assertEqual(dict_run_info, dict(run_info))

    def test_log_metric_allows_multiple_values_at_same_step_and_run_data_uses_max_step_value(self):
        fs = FileStore(self.test_root)
        run_id = self._create_run(fs).info.run_id

        metric_name = "test-metric-1"
        # Check that we get the max of (step, timestamp, value) in that order
        tuples_to_log = [
            (0, 100, 1000),
            (3, 40, 100),  # larger step wins even though it has smaller value
            (3, 50, 10),  # larger timestamp wins even though it has smaller value
            (3, 50, 20),  # tiebreak by max value
            (3, 50, 20),  # duplicate metrics with same (step, timestamp, value) are ok
            # verify that we can log steps out of order / negative steps
            (-3, 900, 900),
            (-1, 800, 800),
        ]
        for step, timestamp, value in reversed(tuples_to_log):
            fs.log_metric(run_id, Metric(metric_name, value, timestamp, step))

        metric_history = fs.get_metric_history(run_id, metric_name)
        logged_tuples = [(m.step, m.timestamp, m.value) for m in metric_history]
        assert set(logged_tuples) == set(tuples_to_log)

        run_data = fs.get_run(run_id).data
        run_metrics = run_data.metrics
        assert len(run_metrics) == 1
        assert run_metrics[metric_name] == 20
        metric_obj = run_data._metric_objs[0]
        assert metric_obj.key == metric_name
        assert metric_obj.step == 3
        assert metric_obj.timestamp == 50
        assert metric_obj.value == 20

    def test_get_all_metrics(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_id in runs:
                run_info = self.run_data[run_id]
                metrics = fs.get_all_metrics(run_id)
                metrics_dict = run_info.pop("metrics")
                for metric in metrics:
                    expected_timestamp, expected_value = max(metrics_dict[metric.key])
                    self.assertEqual(metric.timestamp, expected_timestamp)
                    self.assertEqual(metric.value, expected_value)

    def test_get_metric_history(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_id in runs:
                run_info = self.run_data[run_id]
                metrics = run_info.pop("metrics")
                for metric_name, values in metrics.items():
                    metric_history = fs.get_metric_history(run_id, metric_name)
                    sorted_values = sorted(values, reverse=True)
                    for metric in metric_history:
                        timestamp, metric_value = sorted_values.pop()
                        self.assertEqual(metric.timestamp, timestamp)
                        self.assertEqual(metric.key, metric_name)
                        self.assertEqual(metric.value, metric_value)

    def _search(
        self,
        fs,
        experiment_id,
        filter_str=None,
        run_view_type=ViewType.ALL,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
    ):
        return [
            r.info.run_id
            for r in fs.search_runs([experiment_id], filter_str, run_view_type, max_results)
        ]

    def test_search_runs(self):
        # replace with test with code is implemented
        fs = FileStore(self.test_root)
        # Expect 2 runs for each experiment
        assert len(self._search(fs, self.experiments[0], run_view_type=ViewType.ACTIVE_ONLY)) == 2
        assert len(self._search(fs, self.experiments[0])) == 2
        assert len(self._search(fs, self.experiments[0], run_view_type=ViewType.DELETED_ONLY)) == 0

    def test_search_tags(self):
        fs = FileStore(self.test_root)
        experiment_id = self.experiments[0]
        r1 = fs.create_run(experiment_id, "user", 0, []).info.run_id
        r2 = fs.create_run(experiment_id, "user", 0, []).info.run_id

        fs.set_tag(r1, RunTag("generic_tag", "p_val"))
        fs.set_tag(r2, RunTag("generic_tag", "p_val"))

        fs.set_tag(r1, RunTag("generic_2", "some value"))
        fs.set_tag(r2, RunTag("generic_2", "another value"))

        fs.set_tag(r1, RunTag("p_a", "abc"))
        fs.set_tag(r2, RunTag("p_b", "ABC"))

        # test search returns both runs
        self.assertCountEqual(
            [r1, r2], self._search(fs, experiment_id, filter_str="tags.generic_tag = 'p_val'")
        )
        # test search returns appropriate run (same key different values per run)
        self.assertCountEqual(
            [r1], self._search(fs, experiment_id, filter_str="tags.generic_2 = 'some value'")
        )
        self.assertCountEqual(
            [r2], self._search(fs, experiment_id, filter_str="tags.generic_2='another value'")
        )
        self.assertCountEqual(
            [], self._search(fs, experiment_id, filter_str="tags.generic_tag = 'wrong_val'")
        )
        self.assertCountEqual(
            [], self._search(fs, experiment_id, filter_str="tags.generic_tag != 'p_val'")
        )
        self.assertCountEqual(
            [r1, r2],
            self._search(fs, experiment_id, filter_str="tags.generic_tag != 'wrong_val'"),
        )
        self.assertCountEqual(
            [r1, r2],
            self._search(fs, experiment_id, filter_str="tags.generic_2 != 'wrong_val'"),
        )
        self.assertCountEqual([r1], self._search(fs, experiment_id, filter_str="tags.p_a = 'abc'"))
        self.assertCountEqual([r2], self._search(fs, experiment_id, filter_str="tags.p_b = 'ABC'"))

        self.assertCountEqual(
            [r2], self._search(fs, experiment_id, filter_str="tags.generic_2 LIKE '%other%'")
        )
        self.assertCountEqual(
            [], self._search(fs, experiment_id, filter_str="tags.generic_2 LIKE 'other%'")
        )
        self.assertCountEqual(
            [], self._search(fs, experiment_id, filter_str="tags.generic_2 LIKE '%other'")
        )
        self.assertCountEqual(
            [r2], self._search(fs, experiment_id, filter_str="tags.generic_2 ILIKE '%OTHER%'")
        )

    def test_search_with_max_results(self):
        fs = FileStore(self.test_root)
        exp = fs.create_experiment("search_with_max_results")

        runs = [fs.create_run(exp, "user", r, []).info.run_id for r in range(10)]
        runs.reverse()

        print(runs)
        print(self._search(fs, exp))
        assert runs[:10] == self._search(fs, exp)
        for n in [0, 1, 2, 4, 8, 10, 20, 50, 100, 500, 1000, 1200, 2000]:
            assert runs[: min(1200, n)] == self._search(fs, exp, max_results=n)

        with self.assertRaises(MlflowException) as e:
            self._search(fs, exp, None, max_results=int(1e10))
        self.assertIn("Invalid value for request parameter max_results. It ", e.exception.message)

    def test_search_with_deterministic_max_results(self):
        fs = FileStore(self.test_root)
        exp = fs.create_experiment("test_search_with_deterministic_max_results")

        # Create 10 runs with the same start_time.
        # Sort based on run_id
        runs = sorted([fs.create_run(exp, "user", 1000, []).info.run_id for r in range(10)])
        for n in [0, 1, 2, 4, 8, 10, 20]:
            assert runs[: min(10, n)] == self._search(fs, exp, max_results=n)

    def test_search_runs_pagination(self):
        fs = FileStore(self.test_root)
        exp = fs.create_experiment("test_search_runs_pagination")
        # test returned token behavior
        runs = sorted([fs.create_run(exp, "user", 1000, []).info.run_id for r in range(10)])
        result = fs.search_runs([exp], None, ViewType.ALL, max_results=4)
        assert [r.info.run_id for r in result] == runs[0:4]
        assert result.token is not None
        result = fs.search_runs([exp], None, ViewType.ALL, max_results=4, page_token=result.token)
        assert [r.info.run_id for r in result] == runs[4:8]
        assert result.token is not None
        result = fs.search_runs([exp], None, ViewType.ALL, max_results=4, page_token=result.token)
        assert [r.info.run_id for r in result] == runs[8:]
        assert result.token is None

    def test_weird_param_names(self):
        WEIRD_PARAM_NAME = "this is/a weird/but valid param"
        fs = FileStore(self.test_root)
        run_id = self.exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        fs.log_param(run_id, Param(WEIRD_PARAM_NAME, "Value"))
        run = fs.get_run(run_id)
        assert run.data.params[WEIRD_PARAM_NAME] == "Value"

    def test_log_param_empty_str(self):
        PARAM_NAME = "new param"
        fs = FileStore(self.test_root)
        run_id = self.exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        fs.log_param(run_id, Param(PARAM_NAME, ""))
        run = fs.get_run(run_id)
        assert run.data.params[PARAM_NAME] == ""

    def test_log_param_with_newline(self):
        param_name = "new param"
        param_value = "a string\nwith multiple\nlines"
        fs = FileStore(self.test_root)
        run_id = self.exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        fs.log_param(run_id, Param(param_name, param_value))
        run = fs.get_run(run_id)
        assert run.data.params[param_name] == param_value

    def test_log_param_enforces_value_immutability(self):
        param_name = "new param"
        fs = FileStore(self.test_root)
        run_id = self.exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        fs.log_param(run_id, Param(param_name, "value1"))
        # Duplicate calls to `log_param` with the same key and value should succeed
        fs.log_param(run_id, Param(param_name, "value1"))
        with self.assertRaisesRegex(
            MlflowException, "Changing param values is not allowed. Param with key="
        ) as e:
            fs.log_param(run_id, Param(param_name, "value2"))
        assert e.exception.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        run = fs.get_run(run_id)
        assert run.data.params[param_name] == "value1"

    def test_weird_metric_names(self):
        WEIRD_METRIC_NAME = "this is/a weird/but valid metric"
        fs = FileStore(self.test_root)
        run_id = self.exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        fs.log_metric(run_id, Metric(WEIRD_METRIC_NAME, 10, 1234, 0))
        run = fs.get_run(run_id)
        assert run.data.metrics[WEIRD_METRIC_NAME] == 10
        history = fs.get_metric_history(run_id, WEIRD_METRIC_NAME)
        assert len(history) == 1
        metric = history[0]
        assert metric.key == WEIRD_METRIC_NAME
        assert metric.value == 10
        assert metric.timestamp == 1234

    def test_weird_tag_names(self):
        WEIRD_TAG_NAME = "this is/a weird/but valid tag"
        fs = FileStore(self.test_root)
        run_id = self.exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        fs.set_tag(run_id, RunTag(WEIRD_TAG_NAME, "Muhahaha!"))
        run = fs.get_run(run_id)
        assert run.data.tags[WEIRD_TAG_NAME] == "Muhahaha!"

    def test_set_experiment_tags(self):
        fs = FileStore(self.test_root)
        fs.set_experiment_tag(FileStore.DEFAULT_EXPERIMENT_ID, ExperimentTag("tag0", "value0"))
        fs.set_experiment_tag(FileStore.DEFAULT_EXPERIMENT_ID, ExperimentTag("tag1", "value1"))
        experiment = fs.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
        assert len(experiment.tags) == 2
        assert experiment.tags["tag0"] == "value0"
        assert experiment.tags["tag1"] == "value1"
        # test that updating a tag works
        fs.set_experiment_tag(FileStore.DEFAULT_EXPERIMENT_ID, ExperimentTag("tag0", "value00000"))
        experiment = fs.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
        assert experiment.tags["tag0"] == "value00000"
        assert experiment.tags["tag1"] == "value1"
        # test that setting a tag on 1 experiment does not impact another experiment.
        exp_id = None
        for exp in self.experiments:
            if exp != FileStore.DEFAULT_EXPERIMENT_ID:
                exp_id = exp
                break
        experiment = fs.get_experiment(exp_id)
        assert len(experiment.tags) == 0
        # setting a tag on different experiments maintains different values across experiments
        fs.set_experiment_tag(exp_id, ExperimentTag("tag1", "value11111"))
        experiment = fs.get_experiment(exp_id)
        assert len(experiment.tags) == 1
        assert experiment.tags["tag1"] == "value11111"
        experiment = fs.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
        assert experiment.tags["tag0"] == "value00000"
        assert experiment.tags["tag1"] == "value1"
        # test can set multi-line tags
        fs.set_experiment_tag(exp_id, ExperimentTag("multiline_tag", "value2\nvalue2\nvalue2"))
        experiment = fs.get_experiment(exp_id)
        assert experiment.tags["multiline_tag"] == "value2\nvalue2\nvalue2"
        # test cannot set tags on deleted experiments
        fs.delete_experiment(exp_id)
        with pytest.raises(MlflowException, match="must be in the 'active'lifecycle_stage"):
            fs.set_experiment_tag(exp_id, ExperimentTag("should", "notset"))

    def test_set_tags(self):
        fs = FileStore(self.test_root)
        run_id = self.exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        fs.set_tag(run_id, RunTag("tag0", "value0"))
        fs.set_tag(run_id, RunTag("tag1", "value1"))
        tags = fs.get_run(run_id).data.tags
        assert tags["tag0"] == "value0"
        assert tags["tag1"] == "value1"

        # Can overwrite tags.
        fs.set_tag(run_id, RunTag("tag0", "value2"))
        tags = fs.get_run(run_id).data.tags
        assert tags["tag0"] == "value2"
        assert tags["tag1"] == "value1"

        # Can set multiline tags.
        fs.set_tag(run_id, RunTag("multiline_tag", "value2\nvalue2\nvalue2"))
        tags = fs.get_run(run_id).data.tags
        assert tags["multiline_tag"] == "value2\nvalue2\nvalue2"

    def test_delete_tags(self):
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]["runs"][0]
        fs.set_tag(run_id, RunTag("tag0", "value0"))
        fs.set_tag(run_id, RunTag("tag1", "value1"))
        tags = fs.get_run(run_id).data.tags
        assert tags["tag0"] == "value0"
        assert tags["tag1"] == "value1"
        fs.delete_tag(run_id, "tag0")
        new_tags = fs.get_run(run_id).data.tags
        assert "tag0" not in new_tags.keys()
        # test that you cannot delete tags that don't exist.
        with pytest.raises(MlflowException, match="No tag with name"):
            fs.delete_tag(run_id, "fakeTag")
        # test that you cannot delete tags for nonexistent runs
        with pytest.raises(MlflowException, match=r"Run .+ not found"):
            fs.delete_tag("random_id", "tag0")
        fs = FileStore(self.test_root)
        fs.delete_run(run_id)
        # test that you cannot delete tags for deleted runs.
        assert fs.get_run(run_id).info.lifecycle_stage == LifecycleStage.DELETED
        with pytest.raises(MlflowException, match="must be in 'active' lifecycle_stage"):
            fs.delete_tag(run_id, "tag0")

    def test_unicode_tag(self):
        fs = FileStore(self.test_root)
        run_id = self.exp_data[FileStore.DEFAULT_EXPERIMENT_ID]["runs"][0]
        value = "𝐼 𝓈𝑜𝓁𝑒𝓂𝓃𝓁𝓎 𝓈𝓌𝑒𝒶𝓇 𝓉𝒽𝒶𝓉 𝐼 𝒶𝓂 𝓊𝓅 𝓉𝑜 𝓃𝑜 𝑔𝑜𝑜𝒹"
        fs.set_tag(run_id, RunTag("message", value))
        tags = fs.get_run(run_id).data.tags
        assert tags["message"] == value

    def test_get_deleted_run(self):
        """
        Getting metrics/tags/params/run info should be allowed on deleted runs.
        """
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]["runs"][0]
        fs.delete_run(run_id)
        assert fs.get_run(run_id)

    def test_set_deleted_run(self):
        """
        Setting metrics/tags/params/updating run info should not be allowed on deleted runs.
        """
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]["runs"][0]
        fs.delete_run(run_id)

        assert fs.get_run(run_id).info.lifecycle_stage == LifecycleStage.DELETED
        match = "must be in 'active' lifecycle_stage"
        with pytest.raises(MlflowException, match=match):
            fs.set_tag(run_id, RunTag("a", "b"))
        with pytest.raises(MlflowException, match=match):
            fs.log_metric(run_id, Metric("a", 0.0, timestamp=0, step=0))
        with pytest.raises(MlflowException, match=match):
            fs.log_param(run_id, Param("a", "b"))

    def test_default_experiment_initialization(self):
        fs = FileStore(self.test_root)
        fs.delete_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
        fs = FileStore(self.test_root)
        experiment = fs.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
        assert experiment.lifecycle_stage == LifecycleStage.DELETED

    def test_malformed_experiment(self):
        fs = FileStore(self.test_root)
        exp_0 = fs.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
        assert exp_0.experiment_id == FileStore.DEFAULT_EXPERIMENT_ID

        experiments = len(fs.list_experiments(ViewType.ALL))

        # delete metadata file.
        path = os.path.join(self.test_root, str(exp_0.experiment_id), "meta.yaml")
        os.remove(path)
        with pytest.raises(MissingConfigException, match="does not exist") as e:
            fs.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
            assert e.message.contains("does not exist")

        assert len(fs.list_experiments(ViewType.ALL)) == experiments - 1

    def test_malformed_run(self):
        fs = FileStore(self.test_root)
        exp_0 = fs.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
        all_runs = self._search(fs, exp_0.experiment_id)

        all_run_ids = self.exp_data[exp_0.experiment_id]["runs"]
        assert len(all_runs) == len(all_run_ids)

        # delete metadata file.
        bad_run_id = self.exp_data[exp_0.experiment_id]["runs"][0]
        path = os.path.join(self.test_root, str(exp_0.experiment_id), str(bad_run_id), "meta.yaml")
        os.remove(path)
        with pytest.raises(MissingConfigException, match="does not exist"):
            fs.get_run(bad_run_id)

        valid_runs = self._search(fs, exp_0.experiment_id)
        assert len(valid_runs) == len(all_runs) - 1

        for rid in all_run_ids:
            if rid != bad_run_id:
                fs.get_run(rid)

    def test_mismatching_experiment_id(self):
        fs = FileStore(self.test_root)
        exp_0 = fs.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
        assert exp_0.experiment_id == FileStore.DEFAULT_EXPERIMENT_ID

        experiments = len(fs.list_experiments(ViewType.ALL))

        # mv experiment folder
        target = "1"
        path_orig = os.path.join(self.test_root, str(exp_0.experiment_id))
        path_new = os.path.join(self.test_root, str(target))
        os.rename(path_orig, path_new)

        with pytest.raises(MlflowException, match="Could not find experiment with ID"):
            fs.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)

        with pytest.raises(MlflowException, match="does not exist"):
            fs.get_experiment(target)
        assert len(fs.list_experiments(ViewType.ALL)) == experiments - 1

    def test_bad_experiment_id_recorded_for_run(self):
        fs = FileStore(self.test_root)
        exp_0 = fs.get_experiment(FileStore.DEFAULT_EXPERIMENT_ID)
        all_runs = self._search(fs, exp_0.experiment_id)

        all_run_ids = self.exp_data[exp_0.experiment_id]["runs"]
        assert len(all_runs) == len(all_run_ids)

        # change experiment pointer in run
        bad_run_id = str(self.exp_data[exp_0.experiment_id]["runs"][0])
        path = os.path.join(self.test_root, str(exp_0.experiment_id), bad_run_id)
        experiment_data = read_yaml(path, "meta.yaml")
        experiment_data["experiment_id"] = 1
        write_yaml(path, "meta.yaml", experiment_data, True)

        with pytest.raises(MlflowException, match="metadata is in invalid state"):
            fs.get_run(bad_run_id)

        valid_runs = self._search(fs, exp_0.experiment_id)
        assert len(valid_runs) == len(all_runs) - 1

        for rid in all_run_ids:
            if rid != bad_run_id:
                fs.get_run(rid)

    def test_log_batch(self):
        fs = FileStore(self.test_root)
        run = fs.create_run(
            experiment_id=FileStore.DEFAULT_EXPERIMENT_ID, user_id="user", start_time=0, tags=[]
        )
        run_id = run.info.run_id
        metric_entities = [Metric("m1", 0.87, 12345, 0), Metric("m2", 0.49, 12345, 0)]
        param_entities = [Param("p1", "p1val"), Param("p2", "p2val")]
        tag_entities = [RunTag("t1", "t1val"), RunTag("t2", "t2val")]
        fs.log_batch(
            run_id=run_id, metrics=metric_entities, params=param_entities, tags=tag_entities
        )
        self._verify_logged(fs, run_id, metric_entities, param_entities, tag_entities)

    def _create_run(self, fs):
        return fs.create_run(
            experiment_id=FileStore.DEFAULT_EXPERIMENT_ID, user_id="user", start_time=0, tags=[]
        )

    def test_log_batch_internal_error(self):
        # Verify that internal errors during log_batch result in MlflowExceptions
        fs = FileStore(self.test_root)
        run = self._create_run(fs)

        def _raise_exception_fn(*args, **kwargs):  # pylint: disable=unused-argument
            raise Exception("Some internal error")

        with mock.patch(
            FILESTORE_PACKAGE + ".FileStore._log_run_metric"
        ) as log_metric_mock, mock.patch(
            FILESTORE_PACKAGE + ".FileStore._log_run_param"
        ) as log_param_mock, mock.patch(
            FILESTORE_PACKAGE + ".FileStore._set_run_tag"
        ) as set_tag_mock:
            log_metric_mock.side_effect = _raise_exception_fn
            log_param_mock.side_effect = _raise_exception_fn
            set_tag_mock.side_effect = _raise_exception_fn
            for kwargs in [
                {"metrics": [Metric("a", 3, 1, 0)]},
                {"params": [Param("b", "c")]},
                {"tags": [RunTag("c", "d")]},
            ]:
                log_batch_kwargs = {"metrics": [], "params": [], "tags": []}
                log_batch_kwargs.update(kwargs)
                print(log_batch_kwargs)
                with self.assertRaises(MlflowException) as e:
                    fs.log_batch(run.info.run_id, **log_batch_kwargs)
                self.assertIn(str(e.exception.message), "Some internal error")
                assert e.exception.error_code == ErrorCode.Name(INTERNAL_ERROR)

    def test_log_batch_nonexistent_run(self):
        fs = FileStore(self.test_root)
        nonexistent_uuid = uuid.uuid4().hex
        with self.assertRaises(MlflowException) as e:
            fs.log_batch(nonexistent_uuid, [], [], [])
        assert e.exception.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
        assert ("Run '%s' not found" % nonexistent_uuid) in e.exception.message

    def test_log_batch_params_idempotency(self):
        fs = FileStore(self.test_root)
        run = self._create_run(fs)
        params = [Param("p-key", "p-val")]
        fs.log_batch(run.info.run_id, metrics=[], params=params, tags=[])
        fs.log_batch(run.info.run_id, metrics=[], params=params, tags=[])
        self._verify_logged(fs, run.info.run_id, metrics=[], params=params, tags=[])

    def test_log_batch_tags_idempotency(self):
        fs = FileStore(self.test_root)
        run = self._create_run(fs)
        fs.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])
        fs.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")])
        self._verify_logged(
            fs, run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "t-val")]
        )

    def test_log_batch_allows_tag_overwrite(self):
        fs = FileStore(self.test_root)
        run = self._create_run(fs)
        fs.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "val")])
        fs.log_batch(run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")])
        self._verify_logged(
            fs, run.info.run_id, metrics=[], params=[], tags=[RunTag("t-key", "newval")]
        )

    def test_log_batch_same_metric_repeated_single_req(self):
        fs = FileStore(self.test_root)
        run = self._create_run(fs)
        metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
        metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
        fs.log_batch(run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])
        self._verify_logged(fs, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])

    def test_log_batch_same_metric_repeated_multiple_reqs(self):
        fs = FileStore(self.test_root)
        run = self._create_run(fs)
        metric0 = Metric(key="metric-key", value=1, timestamp=2, step=0)
        metric1 = Metric(key="metric-key", value=2, timestamp=3, step=0)
        fs.log_batch(run.info.run_id, params=[], metrics=[metric0], tags=[])
        self._verify_logged(fs, run.info.run_id, params=[], metrics=[metric0], tags=[])
        fs.log_batch(run.info.run_id, params=[], metrics=[metric1], tags=[])
        self._verify_logged(fs, run.info.run_id, params=[], metrics=[metric0, metric1], tags=[])

    def test_log_batch_allows_tag_overwrite_single_req(self):
        fs = FileStore(self.test_root)
        run = self._create_run(fs)
        tags = [RunTag("t-key", "val"), RunTag("t-key", "newval")]
        fs.log_batch(run.info.run_id, metrics=[], params=[], tags=tags)
        self._verify_logged(fs, run.info.run_id, metrics=[], params=[], tags=[tags[-1]])

    def test_log_batch_accepts_empty_payload(self):
        fs = FileStore(self.test_root)
        run = self._create_run(fs)
        fs.log_batch(run.info.run_id, metrics=[], params=[], tags=[])
        self._verify_logged(fs, run.info.run_id, metrics=[], params=[], tags=[])

    def test_log_batch_with_duplicate_params_errors_no_partial_write(self):
        fs = FileStore(self.test_root)
        run = self._create_run(fs)
        with self.assertRaisesRegex(
            MlflowException, "Duplicate parameter keys have been submitted"
        ) as e:
            fs.log_batch(
                run.info.run_id, metrics=[], params=[Param("a", "1"), Param("a", "2")], tags=[]
            )
        assert e.exception.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
        self._verify_logged(fs, run.info.run_id, metrics=[], params=[], tags=[])
