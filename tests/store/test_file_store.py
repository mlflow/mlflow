import os
import shutil
import time
import unittest
import uuid

from mlflow.entities.experiment import Experiment
from mlflow.store.file_store import FileStore
from mlflow.utils.file_utils import write_yaml
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
        except Exception as e:  # noqa
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

    def test_get_run(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            runs = self.exp_data[exp_id]["runs"]
            for run_uuid in runs:
                run = fs.get_run(run_uuid)
                run_info = self.run_data[run_uuid]
                run_info.pop("metrics")
                run_info.pop("params")
                self.assertEqual(run_info, dict(run.info))

    def test_list_run_infos(self):
        fs = FileStore(self.test_root)
        for exp_id in self.experiments:
            run_infos = fs.list_run_infos(exp_id)
            for run_info in run_infos:
                run_uuid = run_info.run_uuid
                dict_run_info = self.run_data[run_uuid]
                dict_run_info.pop("metrics")
                dict_run_info.pop("params")
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

    def test_list_artifacts(self):
        fs = FileStore(self.test_root)
        # replace with test with code is implemented
        with self.assertRaises(Exception):
            self.assertIsNotNone(fs.list_artifacts("random uuid", "some relative path"))

    def test_get_artifact(self):
        fs = FileStore(self.test_root)
        # replace with test with code is implemented
        with self.assertRaises(Exception):
            self.assertIsNotNone(fs.get_artifact("random uuid", "some relative path"))

    def test_search_runs(self):
        # replace with test with code is implemented
        fs = FileStore(self.test_root)
        # Expect 2 runs for each experiment
        assert len(fs.search_runs([self.experiments[0]], [])) == 2
