import os

import uuid

from mlflow.entities.experiment import Experiment
from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run import Run
from mlflow.entities.run_data import RunData
from mlflow.entities.run_info import RunInfo

from mlflow.entities.run_status import RunStatus
from mlflow.store.abstract_store import AbstractStore
from mlflow.utils.validation import _validate_metric_name, _validate_param_name, _validate_run_id

from mlflow.utils.env import get_env
from mlflow.utils.file_utils import (is_directory, list_subdirs, mkdir, exists, write_yaml,
                                     read_yaml, find, read_file, build_path, write_to, append_to,
                                     make_containing_dirs)

from mlflow.utils.search_utils import does_run_match_clause

_TRACKING_DIR_ENV_VAR = "MLFLOW_TRACKING_DIR"


def _default_root_dir():
    return get_env(_TRACKING_DIR_ENV_VAR) or os.path.abspath("mlruns")


class FileStore(AbstractStore):
    ARTIFACTS_FOLDER_NAME = "artifacts"
    METRICS_FOLDER_NAME = "metrics"
    PARAMS_FOLDER_NAME = "params"
    META_DATA_FILE_NAME = "meta.yaml"

    def __init__(self, root_directory=None, artifact_root_uri=None):
        """
        Create a new FileStore with the given root directory and a given default artifact root URI.
        """
        super(FileStore, self).__init__()
        self.root_directory = root_directory or _default_root_dir()
        self.artifact_root_uri = artifact_root_uri or self.root_directory
        # Create root directory if needed
        if not exists(self.root_directory):
            mkdir(self.root_directory)
        # Create default experiment if needed
        if not self._has_experiment(experiment_id=Experiment.DEFAULT_EXPERIMENT_ID):
            self._create_experiment_with_id(name="Default",
                                            experiment_id=Experiment.DEFAULT_EXPERIMENT_ID,
                                            artifact_uri=None)

    def _check_root_dir(self):
        """
        Run checks before running directory operations.
        """
        if not exists(self.root_directory):
            raise Exception("'%s' does not exist." % self.root_directory)
        if not is_directory(self.root_directory):
            raise Exception("'%s' is not a directory." % self.root_directory)

    def _get_experiment_dir(self, experiment_id):
        return build_path(self.root_directory, str(experiment_id))

    def _get_run_dir(self, experiment_id, run_uuid):
        _validate_run_id(run_uuid)
        return build_path(self._get_experiment_dir(experiment_id), run_uuid)

    def _get_metric_path(self, experiment_id, run_uuid, metric_key):
        _validate_run_id(run_uuid)
        _validate_metric_name(metric_key)
        return build_path(self._get_run_dir(experiment_id, run_uuid), FileStore.METRICS_FOLDER_NAME,
                          metric_key)

    def _get_param_path(self, experiment_id, run_uuid, param_name):
        _validate_run_id(run_uuid)
        _validate_param_name(param_name)
        return build_path(self._get_run_dir(experiment_id, run_uuid), FileStore.PARAMS_FOLDER_NAME,
                          param_name)

    def _get_artifact_dir(self, experiment_id, run_uuid):
        _validate_run_id(run_uuid)
        artifacts_dir = build_path(self.get_experiment(experiment_id).artifact_location,
                                   run_uuid,
                                   FileStore.ARTIFACTS_FOLDER_NAME)
        return artifacts_dir

    def list_experiments(self):
        self._check_root_dir()
        return [self.get_experiment(exp_id) for exp_id in list_subdirs(self.root_directory)]

    def _create_experiment_with_id(self, name, experiment_id, artifact_uri):
        self._check_root_dir()
        meta_dir = mkdir(self.root_directory, str(experiment_id))
        artifact_uri = artifact_uri or build_path(self.artifact_root_uri, str(experiment_id))
        experiment = Experiment(experiment_id, name, artifact_uri)
        write_yaml(meta_dir, FileStore.META_DATA_FILE_NAME, dict(experiment))
        return experiment_id

    def create_experiment(self, name, artifact_location=None):
        self._check_root_dir()
        if name is None or name == "":
            raise Exception("Invalid experiment name '%s'" % name)
        experiment = self.get_experiment_by_name(name)
        if experiment is not None:
            raise Exception("Experiment '%s' already exists." % experiment.name)
        # Get all existing experiments and find the one with largest ID.
        # len(list_all(..)) would not work when experiments are deleted.
        experiments_ids = [e.experiment_id for e in self.list_experiments()]
        experiment_id = max(experiments_ids) + 1
        return self._create_experiment_with_id(name, experiment_id, artifact_location)

    def _has_experiment(self, experiment_id):
        return len(find(self.root_directory, str(experiment_id), full_path=True)) > 0

    @staticmethod
    def _get_experiment(experiment_dir_path):
        meta = read_yaml(experiment_dir_path, FileStore.META_DATA_FILE_NAME)
        return Experiment.from_dictionary(meta)

    def get_experiment(self, experiment_id):
        self._check_root_dir()
        experiment_dirs = find(self.root_directory, str(experiment_id), full_path=True)
        if len(experiment_dirs) == 0:
            raise Exception("Could not find experiment with ID %s" % experiment_id)
        return self._get_experiment(experiment_dirs[0])

    def get_experiment_by_name(self, name):
        self._check_root_dir()
        all_experiments = list_subdirs(self.root_directory, full_path=True)
        for experiment_dir in all_experiments:
            experiment = self._get_experiment(experiment_dir)
            if experiment.name == name:
                return experiment
        return None

    def _find_run_root(self, run_uuid):
        _validate_run_id(run_uuid)
        self._check_root_dir()
        all_experiments = list_subdirs(self.root_directory, full_path=True)
        for experiment_dir in all_experiments:
            runs = find(experiment_dir, run_uuid, full_path=True)
            if len(runs) == 0:
                continue
            return runs[0]
        return None

    def update_run_info(self, run_uuid, run_status, end_time):
        _validate_run_id(run_uuid)
        run_info = self.get_run(run_uuid).info
        new_info = run_info.copy_with_overrides(run_status, end_time)
        run_dir = self._get_run_dir(run_info.experiment_id, run_info.run_uuid)
        write_yaml(run_dir, FileStore.META_DATA_FILE_NAME, dict(new_info), overwrite=True)
        return new_info

    def create_run(self, experiment_id, user_id, run_name, source_type,
                   source_name, entry_point_name, start_time, source_version, tags):
        """
        Creates a run with the specified attributes.
        """
        if self.get_experiment(experiment_id) is None:
            raise Exception("Could not create run under experiment with ID %s - no such experiment "
                            "exists." % experiment_id)
        run_uuid = uuid.uuid4().hex
        artifact_uri = self._get_artifact_dir(experiment_id, run_uuid)
        num_runs = len(self._list_run_uuids(experiment_id))
        run_info = RunInfo(run_uuid=run_uuid, experiment_id=experiment_id, name="Run %s" % num_runs,
                           artifact_uri=artifact_uri, source_type=source_type,
                           source_name=source_name,
                           entry_point_name=entry_point_name, user_id=user_id,
                           status=RunStatus.RUNNING, start_time=start_time, end_time=None,
                           source_version=source_version, tags=tags)
        # Persist run metadata and create directories for logging metrics, parameters, artifacts
        run_dir = self._get_run_dir(run_info.experiment_id, run_info.run_uuid)
        mkdir(run_dir)
        write_yaml(run_dir, FileStore.META_DATA_FILE_NAME, dict(run_info))
        mkdir(run_dir, FileStore.METRICS_FOLDER_NAME)
        mkdir(run_dir, FileStore.PARAMS_FOLDER_NAME)
        mkdir(run_dir, FileStore.ARTIFACTS_FOLDER_NAME)
        return Run(run_info=run_info, run_data=None)

    def get_run(self, run_uuid):
        _validate_run_id(run_uuid)
        run_dir = self._find_run_root(run_uuid)
        if run_dir is None:
            raise Exception("Run '%s' not found" % run_uuid)
        run_info = self.get_run_info(run_dir)
        metrics = self.get_all_metrics(run_uuid)
        params = self.get_all_params(run_uuid)
        return Run(run_info, RunData(metrics, params))

    @staticmethod
    def get_run_info(run_dir):
        meta = read_yaml(run_dir, FileStore.META_DATA_FILE_NAME)
        return RunInfo.from_dictionary(meta)

    def _get_run_files(self, run_uuid, resource_type):
        _validate_run_id(run_uuid)
        if resource_type == "metric":
            subfolder_name = FileStore.METRICS_FOLDER_NAME
        elif resource_type == "param":
            subfolder_name = FileStore.PARAMS_FOLDER_NAME
        else:
            raise Exception("Looking for unknown resource under run.")
        run_dir = self._find_run_root(run_uuid)
        if run_dir is None:
            raise Exception("Run '%s' not found" % run_uuid)
        source_dirs = find(run_dir, subfolder_name, full_path=True)
        if len(source_dirs) == 0:
            raise Exception("Malformed run '%s'." % run_uuid)
        file_names = []
        for root, _, files in os.walk(source_dirs[0]):
            for name in files:
                abspath = os.path.join(root, name)
                file_names.append(os.path.relpath(abspath, source_dirs[0]))
        return source_dirs[0], file_names

    @staticmethod
    def _get_metric_from_file(parent_path, metric_name):
        _validate_metric_name(metric_name)
        metric_data = read_file(parent_path, metric_name)
        if len(metric_data) == 0:
            raise Exception("Metric '%s' is malformed. No data found." % metric_name)
        last_line = metric_data[-1]
        timestamp, val = last_line.strip().split(" ")
        return Metric(metric_name, float(val), int(timestamp))

    def get_metric(self, run_uuid, metric_key):
        _validate_run_id(run_uuid)
        _validate_metric_name(metric_key)
        parent_path, metric_files = self._get_run_files(run_uuid, "metric")
        if metric_key not in metric_files:
            raise Exception("Metric '%s' not found under run '%s'" % (metric_key, run_uuid))
        return self._get_metric_from_file(parent_path, metric_key)

    def get_all_metrics(self, run_uuid):
        _validate_run_id(run_uuid)
        parent_path, metric_files = self._get_run_files(run_uuid, "metric")
        metrics = []
        for metric_file in metric_files:
            metrics.append(self._get_metric_from_file(parent_path, metric_file))
        return metrics

    def get_metric_history(self, run_uuid, metric_key):
        _validate_run_id(run_uuid)
        _validate_metric_name(metric_key)
        parent_path, metric_files = self._get_run_files(run_uuid, "metric")
        if metric_key not in metric_files:
            raise Exception("Metric '%s' not found under run '%s'" % (metric_key, run_uuid))
        metric_data = read_file(parent_path, metric_key)
        rsl = []
        for pair in metric_data:
            ts, val = pair.strip().split(" ")
            rsl.append(Metric(metric_key, float(val), int(ts)))
        return rsl

    @staticmethod
    def _get_param_from_file(parent_path, param_name):
        _validate_param_name(param_name)
        param_data = read_file(parent_path, param_name)
        if len(param_data) == 0:
            raise Exception("Param '%s' is malformed. No data found." % param_name)
        if len(param_data) > 1:
            raise Exception("Unexpected data for param '%s'. Param recorded more than once"
                            % param_name)
        return Param(param_name, str(param_data[0].strip()))

    def get_param(self, run_uuid, param_name):
        _validate_run_id(run_uuid)
        _validate_param_name(param_name)
        parent_path, param_files = self._get_run_files(run_uuid, "param")
        if param_name not in param_files:
            raise Exception("Param '%s' not found under run '%s'" % (param_name, run_uuid))
        return self._get_param_from_file(parent_path, param_name)

    def get_all_params(self, run_uuid):
        parent_path, param_files = self._get_run_files(run_uuid, "param")
        params = []
        for param_file in param_files:
            params.append(self._get_param_from_file(parent_path, param_file))
        return params

    def _list_run_uuids(self, experiment_id):
        self._check_root_dir()
        experiment_dir = find(self.root_directory, str(experiment_id), full_path=True)[0]
        return list_subdirs(experiment_dir, full_path=False)

    def search_runs(self, experiment_ids, search_expressions):
        run_uuids = []
        if len(search_expressions) == 0:
            for experiment_id in experiment_ids:
                run_uuids.extend(self._list_run_uuids(experiment_id))
        else:
            for experiment_id in experiment_ids:
                for run_uuid in self._list_run_uuids(experiment_id):
                    run = self.get_run(run_uuid)
                    if all([does_run_match_clause(run, s) for s in search_expressions]):
                        run_uuids.append(run_uuid)
        return [self.get_run(run_uuid) for run_uuid in run_uuids]

    def list_run_infos(self, experiment_id):
        run_infos = []
        for run_uuid in self._list_run_uuids(experiment_id):
            run_infos.append(self.get_run_info(self._get_run_dir(experiment_id, run_uuid)))
        return run_infos

    def log_metric(self, run_uuid, metric):
        _validate_run_id(run_uuid)
        _validate_metric_name(metric.key)
        run = self.get_run(run_uuid)
        metric_path = self._get_metric_path(run.info.experiment_id, run_uuid, metric.key)
        make_containing_dirs(metric_path)
        append_to(metric_path, "%s %s\n" % (metric.timestamp, metric.value))

    def log_param(self, run_uuid, param):
        _validate_run_id(run_uuid)
        _validate_param_name(param.key)
        run = self.get_run(run_uuid)
        param_path = self._get_param_path(run.info.experiment_id, run_uuid, param.key)
        make_containing_dirs(param_path)
        write_to(param_path, "%s\n" % param.value)
