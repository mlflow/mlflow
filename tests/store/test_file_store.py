#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import time
import unittest
import uuid

import mock
import pytest

from mlflow.entities import Experiment, Metric, Param, RunTag, ViewType, RunInfo
from mlflow.exceptions import MlflowException, MissingConfigException
from mlflow.store.file_store import FileStore
from mlflow.utils.file_utils import write_yaml, read_yaml
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from tests.helper_functions import random_int, random_str


class TestFileStore(unittest.TestCase):
    ROOT_LOCATION = "/tmp"

    def setUp(self):
        self._create_root(TestFileStore.ROOT_LOCATION)
        self.maxDiff = None

    def _create_root(self, root):
        self.test_root = os.path.join(root, "test_file_store_%d" % random_int())
        os.mkdir(self.test_root)
        self.experiments = [random_int(100, int(1e9)) for _ in range(3)]
        self.exp_data = {}
        self.run_data = {}
        # Include default experiment
        self.experiments.append(Experiment.DEFAULT_EXPERIMENT_ID)
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
                run_uuid = uuid.uuid4().hex
                self.exp_data[exp]["runs"].append(run_uuid)
                run_folder = os.path.join(exp_folder, run_uuid)
                os.makedirs(run_folder)
                run_info = {"run_uuid": run_uuid,
                            "experiment_id": exp,
                            "name": random_str(random_int(10, 40)),
                            "source_type": random_int(1, 4),
                            "source_name": random_str(random_int(100, 300)),
                            "entry_point_name": random_str(random_int(100, 300)),
                            "user_id": random_str(random_int(10, 25)),
                            "status": random_int(1, 5),
                            "start_time": random_int(1, 10),
                            "end_time": random_int(20, 30),
                            "source_version": random_str(random_int(10, 30)),
                            "tags": [],
                            "artifact_uri": "%s/%s" % (run_folder, FileStore.ARTIFACTS_FOLDER_NAME),
                            }
                write_yaml(run_folder, FileStore.META_DATA_FILE_NAME, run_info)
                self.run_data[run_uuid] = run_info
                # params
                params_folder = os.path.join(run_folder, FileStore.PARAMS_FOLDER_NAME)
                os.makedirs(params_folder)
                params = {}
                for _ in range(5):
                    param_name = random_str(random_int(4, 12))
                    param_value = random_str(random_int(10, 15))
                    param_file = os.path.join(params_folder, param_name)
                    with open(param_file, 'w') as f:
                        f.write(param_value)
                    params[param_name] = param_value
                self.run_data[run_uuid]["params"] = params
                # metrics
                metrics_folder = os.path.join(run_folder, FileStore.METRICS_FOLDER_NAME)
                os.makedirs(metrics_folder)
                metrics = {}
                for _ in range(3):
                    metric_name = random_str(random_int(6, 10))
                    timestamp = int(time.time())
                    metric_file = os.path.join(metrics_folder, metric_name)
                    values = []
                    for _ in range(10):
                        metric_value = random_int(100, 2000)
                        timestamp += random_int(10000, 2000000)
                        values.append((timestamp, metric_value))
                        with open(metric_file, 'a') as f:
                            f.write("%d %d\n" % (timestamp, metric_value))
                    metrics[metric_name] = values
                self.run_data[run_uuid]["metrics"] = metrics
                # artifacts
                os.makedirs(os.path.join(run_folder, FileStore.ARTIFACTS_FOLDER_NAME))

    def tearDown(self):
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_valid_root(self):
        # Test with valid root
        file_store = FileStore(self.test_root)
        try:
            file_store._check_root_dir()
        except Exception as e:  # pylint: disable=broad-except
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

    def test_get_experiment(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            exp = fs.get_experiment(exp_id)
            self.assertEqual(exp.experiment_id, exp_id)
            self.assertEqual(exp.name, self.exp_data[exp_id]["name"])
            self.assertEqual(exp.artifact_location, self.exp_data[exp_id]["artifact_location"])

        # test that fake experiments dont exist.
        # look for random experiment ids between 8000, 15000 since created ones are (100, 2000)
        for exp_id in set(random_int(8000, 15000) for x in range(20)):
            with self.assertRaises(Exception):
                fs.get_experiment(exp_id)

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
        fs.create_experiment(random_str(1))
        fs._create_experiment_with_id.assert_called_once()
        experiment_id = fs._create_experiment_with_id.call_args[0][1]
        self.assertEqual(experiment_id, 0)

    def test_create_experiment(self):
        fs = FileStore(self.test_root)

        # Error cases
        with self.assertRaises(Exception):
            fs.create_experiment(None)
        with self.assertRaises(Exception):
            fs.create_experiment("")

        next_id = max(self.experiments) + 1
        name = random_str(25)  # since existing experiments are 10 chars long
        created_id = fs.create_experiment(name)
        # test that newly created experiment matches expected id
        self.assertEqual(created_id, next_id)

        # get the new experiment (by id) and verify (by name)
        exp1 = fs.get_experiment(created_id)
        self.assertEqual(exp1.name, name)

        # get the new experiment (by name) and verify (by id)
        exp2 = fs.get_experiment_by_name(name)
        self.assertEqual(exp2.experiment_id, created_id)

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
        self.assertEqual(fs.get_experiment(exp_id).lifecycle_stage,
                         Experiment.DELETED_LIFECYCLE)

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
        self.assertEqual(fs.get_experiment(exp_id).lifecycle_stage,
                         Experiment.ACTIVE_LIFECYCLE)

    def test_rename_experiment(self):
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        exp_name = self.exp_data[exp_id]["name"]
        new_name = exp_name + "!!!"
        self.assertNotEqual(exp_name, new_name)
        self.assertEqual(fs.get_experiment(exp_id).name, exp_name)
        fs.rename_experiment(exp_id, new_name)
        self.assertEqual(fs.get_experiment(exp_id).name, new_name)

        # Ensure that we cannot rename deleted experiments.
        fs.delete_experiment(exp_id)
        with pytest.raises(Exception) as e:
            fs.rename_experiment(exp_id, exp_name)
        assert 'non-active lifecycle' in str(e.value)
        self.assertEqual(fs.get_experiment(exp_id).name, new_name)

        # Restore the experiment, and confirm that we acn now rename it.
        fs.restore_experiment(exp_id)
        self.assertEqual(fs.get_experiment(exp_id).name, new_name)
        fs.rename_experiment(exp_id, exp_name)
        self.assertEqual(fs.get_experiment(exp_id).name, exp_name)

    def test_delete_restore_run(self):
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]['runs'][0]
        # Should not throw.
        assert fs.get_run(run_id).info.lifecycle_stage == 'active'
        fs.delete_run(run_id)
        assert fs.get_run(run_id).info.lifecycle_stage == 'deleted'
        fs.restore_run(run_id)
        assert fs.get_run(run_id).info.lifecycle_stage == 'active'

    def test_create_run_in_deleted_experiment(self):
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        # delete it
        fs.delete_experiment(exp_id)
        with pytest.raises(Exception):
            fs.create_run(exp_id, 'user', 'name', 'source_type', 'source_name', 'entry_point_name',
                          0, None, [], None)

    def test_get_run(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_uuid in runs:
                run = fs.get_run(run_uuid)
                run_info = self.run_data[run_uuid]
                run_info.pop("metrics")
                run_info.pop("params")
                run_info.pop("tags")
                run_info['lifecycle_stage'] = RunInfo.ACTIVE_LIFECYCLE
                self.assertEqual(run_info, dict(run.info))

    def test_list_run_infos(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            run_infos = fs.list_run_infos(exp_id, run_view_type=ViewType.ALL)
            for run_info in run_infos:
                run_uuid = run_info.run_uuid
                dict_run_info = self.run_data[run_uuid]
                dict_run_info.pop("metrics")
                dict_run_info.pop("params")
                dict_run_info.pop("tags")
                dict_run_info['lifecycle_stage'] = RunInfo.ACTIVE_LIFECYCLE
                self.assertEqual(dict_run_info, dict(run_info))

    def test_get_metric(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_uuid in runs:
                run_info = self.run_data[run_uuid]
                metrics_dict = run_info.pop("metrics")
                for metric_name, values in metrics_dict.items():
                    # just the last recorded value
                    timestamp, metric_value = values[-1]
                    metric = fs.get_metric(run_uuid, metric_name)
                    self.assertEqual(metric.timestamp, timestamp)
                    self.assertEqual(metric.key, metric_name)
                    self.assertEqual(metric.value, metric_value)

    def test_get_all_metrics(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_uuid in runs:
                run_info = self.run_data[run_uuid]
                metrics = fs.get_all_metrics(run_uuid)
                metrics_dict = run_info.pop("metrics")
                for metric in metrics:
                    # just the last recorded value
                    timestamp, metric_value = metrics_dict[metric.key][-1]
                    self.assertEqual(metric.timestamp, timestamp)
                    self.assertEqual(metric.value, metric_value)

    def test_get_metric_history(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_uuid in runs:
                run_info = self.run_data[run_uuid]
                metrics = run_info.pop("metrics")
                for metric_name, values in metrics.items():
                    metric_history = fs.get_metric_history(run_uuid, metric_name)
                    sorted_values = sorted(values, reverse=True)
                    for metric in metric_history:
                        timestamp, metric_value = sorted_values.pop()
                        self.assertEqual(metric.timestamp, timestamp)
                        self.assertEqual(metric.key, metric_name)
                        self.assertEqual(metric.value, metric_value)

    def test_get_param(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_uuid in runs:
                run_info = self.run_data[run_uuid]
                params_dict = run_info.pop("params")
                for param_name, param_value in params_dict.items():
                    param = fs.get_param(run_uuid, param_name)
                    self.assertEqual(param.key, param_name)
                    self.assertEqual(param.value, param_value)

    def test_search_runs(self):
        # replace with test with code is implemented
        fs = FileStore(self.test_root)
        # Expect 2 runs for each experiment
        assert len(fs.search_runs([self.experiments[0]], [], run_view_type=ViewType.ACTIVE_ONLY)) \
            == 2
        assert len(fs.search_runs([self.experiments[0]], [], run_view_type=ViewType.ALL)) \
            == 2
        assert len(fs.search_runs([self.experiments[0]], [], run_view_type=ViewType.DELETED_ONLY)) \
            == 0

    def test_weird_param_names(self):
        WEIRD_PARAM_NAME = "this is/a weird/but valid param"
        fs = FileStore(self.test_root)
        run_uuid = self.exp_data[0]["runs"][0]
        fs.log_param(run_uuid, Param(WEIRD_PARAM_NAME, "Value"))
        param = fs.get_param(run_uuid, WEIRD_PARAM_NAME)
        assert param.key == WEIRD_PARAM_NAME
        assert param.value == "Value"

    def test_weird_metric_names(self):
        WEIRD_METRIC_NAME = "this is/a weird/but valid metric"
        fs = FileStore(self.test_root)
        run_uuid = self.exp_data[0]["runs"][0]
        fs.log_metric(run_uuid, Metric(WEIRD_METRIC_NAME, 10, 1234))
        metric = fs.get_metric(run_uuid, WEIRD_METRIC_NAME)
        assert metric.key == WEIRD_METRIC_NAME
        assert metric.value == 10
        assert metric.timestamp == 1234

    def test_weird_tag_names(self):
        WEIRD_TAG_NAME = "this is/a weird/but valid tag"
        fs = FileStore(self.test_root)
        run_uuid = self.exp_data[0]["runs"][0]
        fs.set_tag(run_uuid, RunTag(WEIRD_TAG_NAME, "Muhahaha!"))
        tag = fs.get_run(run_uuid).data.tags[0]
        assert tag.key == WEIRD_TAG_NAME
        assert tag.value == "Muhahaha!"

    def test_set_tags(self):
        fs = FileStore(self.test_root)
        run_uuid = self.exp_data[0]["runs"][0]
        fs.set_tag(run_uuid, RunTag("tag0", "value0"))
        fs.set_tag(run_uuid, RunTag("tag1", "value1"))
        tags = [(t.key, t.value) for t in fs.get_run(run_uuid).data.tags]
        assert set(tags) == {
            ("tag0", "value0"),
            ("tag1", "value1"),
        }

        # Can overwrite tags.
        fs.set_tag(run_uuid, RunTag("tag0", "value2"))
        tags = [(t.key, t.value) for t in fs.get_run(run_uuid).data.tags]
        assert set(tags) == {
            ("tag0", "value2"),
            ("tag1", "value1"),
        }

        # Can set multiline tags.
        fs.set_tag(run_uuid, RunTag("multiline_tag", "value2\nvalue2\nvalue2"))
        tags = [(t.key, t.value) for t in fs.get_run(run_uuid).data.tags]
        assert set(tags) == {
            ("tag0", "value2"),
            ("tag1", "value1"),
            ("multiline_tag", "value2\nvalue2\nvalue2"),
        }

    def test_unicode_tag(self):
        fs = FileStore(self.test_root)
        run_uuid = self.exp_data[0]["runs"][0]
        value = u"𝐼 𝓈𝑜𝓁𝑒𝓂𝓃𝓁𝓎 𝓈𝓌𝑒𝒶𝓇 𝓉𝒽𝒶𝓉 𝐼 𝒶𝓂 𝓊𝓅 𝓉𝑜 𝓃𝑜 𝑔𝑜𝑜𝒹"
        fs.set_tag(run_uuid, RunTag("message", value))
        tag = fs.get_run(run_uuid).data.tags[0]
        assert tag.key == "message"
        assert tag.value == value

    def test_get_deleted_run(self):
        """
        Getting metrics/tags/params/run info should be allowed on deleted runs.
        """
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]['runs'][0]
        fs.delete_run(run_id)

        run = fs.get_run(run_id)
        assert fs.get_metric(run_id, run.data.metrics[0].key).value == run.data.metrics[0].value
        assert fs.get_param(run_id, run.data.params[0].key).value == run.data.params[0].value

    def test_set_deleted_run(self):
        """
        Setting metrics/tags/params/updating run info should not be allowed on deleted runs.
        """
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run_id = self.exp_data[exp_id]['runs'][0]
        fs.delete_run(run_id)

        assert fs.get_run(run_id).info.lifecycle_stage == RunInfo.DELETED_LIFECYCLE
        with pytest.raises(MlflowException):
            fs.set_tag(run_id, RunTag('a', 'b'))
        with pytest.raises(MlflowException):
            fs.log_metric(run_id, Metric('a', 0.0, timestamp=0))
        with pytest.raises(MlflowException):
            fs.log_param(run_id, Param('a', 'b'))

    def test_create_run_with_parent_id(self):
        fs = FileStore(self.test_root)
        exp_id = self.experiments[random_int(0, len(self.experiments) - 1)]
        run = fs.create_run(exp_id, 'user', 'name', 'source_type', 'source_name',
                            'entry_point_name', 0, None, [], 'test_parent_run_id')
        assert any([t.key == MLFLOW_PARENT_RUN_ID and t.value == 'test_parent_run_id'
                    for t in fs.get_all_tags(run.info.run_uuid)])

    def test_default_experiment_initialization(self):
        fs = FileStore(self.test_root)
        fs.delete_experiment(Experiment.DEFAULT_EXPERIMENT_ID)
        fs = FileStore(self.test_root)
        assert fs.get_experiment(0).lifecycle_stage == Experiment.DELETED_LIFECYCLE

    def test_malformed_experiment(self):
        fs = FileStore(self.test_root)
        exp_0 = fs.get_experiment(Experiment.DEFAULT_EXPERIMENT_ID)
        assert exp_0.experiment_id == Experiment.DEFAULT_EXPERIMENT_ID

        experiments = len(fs.list_experiments(ViewType.ALL))

        # delete metadata file.
        path = os.path.join(self.test_root, str(exp_0.experiment_id), "meta.yaml")
        os.remove(path)
        with pytest.raises(MissingConfigException) as e:
            fs.get_experiment(Experiment.DEFAULT_EXPERIMENT_ID)
            assert e.message.contains("does not exist")

        assert len(fs.list_experiments(ViewType.ALL)) == experiments - 1

    def test_malformed_run(self):
        fs = FileStore(self.test_root)
        exp_0 = fs.get_experiment(Experiment.DEFAULT_EXPERIMENT_ID)
        all_runs = fs.search_runs([exp_0.experiment_id], [], run_view_type=ViewType.ALL)

        all_run_ids = self.exp_data[exp_0.experiment_id]["runs"]
        assert len(all_runs) == len(all_run_ids)

        # delete metadata file.
        bad_run_id = self.exp_data[exp_0.experiment_id]['runs'][0]
        path = os.path.join(self.test_root, str(exp_0.experiment_id), str(bad_run_id), "meta.yaml")
        os.remove(path)
        with pytest.raises(MissingConfigException) as e:
            fs.get_run(bad_run_id)
            assert e.message.contains("does not exist")

        valid_runs = fs.search_runs([exp_0.experiment_id], [], run_view_type=ViewType.ALL)
        assert len(valid_runs) == len(all_runs) - 1

        for rid in all_run_ids:
            if rid != bad_run_id:
                fs.get_run(rid)

    def test_mismatching_experiment_id(self):
        fs = FileStore(self.test_root)
        exp_0 = fs.get_experiment(Experiment.DEFAULT_EXPERIMENT_ID)
        assert exp_0.experiment_id == Experiment.DEFAULT_EXPERIMENT_ID

        experiments = len(fs.list_experiments(ViewType.ALL))

        # mv experiment folder
        target = 1
        path_orig = os.path.join(self.test_root, str(exp_0.experiment_id))
        path_new = os.path.join(self.test_root, str(target))
        os.rename(path_orig, path_new)

        with pytest.raises(MlflowException) as e:
            fs.get_experiment(Experiment.DEFAULT_EXPERIMENT_ID)
            assert e.message.contains("Could not find experiment with ID")

        with pytest.raises(MlflowException) as e:
            fs.get_experiment(target)
            assert e.message.contains("does not exist")
        assert len(fs.list_experiments(ViewType.ALL)) == experiments - 1

    def test_bad_experiment_id_recorded_for_run(self):
        fs = FileStore(self.test_root)
        exp_0 = fs.get_experiment(Experiment.DEFAULT_EXPERIMENT_ID)
        all_runs = fs.search_runs([exp_0.experiment_id], [], run_view_type=ViewType.ALL)

        all_run_ids = self.exp_data[exp_0.experiment_id]["runs"]
        assert len(all_runs) == len(all_run_ids)

        # change experiment pointer in run
        bad_run_id = str(self.exp_data[exp_0.experiment_id]['runs'][0])
        path = os.path.join(self.test_root, str(exp_0.experiment_id), bad_run_id)
        experiment_data = read_yaml(path, "meta.yaml")
        experiment_data["experiment_id"] = 1
        write_yaml(path, "meta.yaml", experiment_data, True)

        with pytest.raises(MlflowException) as e:
            fs.get_run(bad_run_id)
            assert e.message.contains("not found")

        valid_runs = fs.search_runs([exp_0.experiment_id], [], run_view_type=ViewType.ALL)
        assert len(valid_runs) == len(all_runs) - 1

        for rid in all_run_ids:
            if rid != bad_run_id:
                fs.get_run(rid)
