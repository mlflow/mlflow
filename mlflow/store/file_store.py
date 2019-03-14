import logging
import os

import uuid
import six

from mlflow.entities import Experiment, Metric, Param, Run, RunData, RunInfo, RunStatus, RunTag, \
                            ViewType
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.run_info import check_run_is_active, check_run_is_deleted
from mlflow.exceptions import MlflowException, MissingConfigException
import mlflow.protos.databricks_pb2 as databricks_pb2
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.store import DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH
from mlflow.store.abstract_store import AbstractStore
from mlflow.utils.validation import _validate_metric_name, _validate_param_name, _validate_run_id, \
                                    _validate_tag_name, _validate_experiment_id

from mlflow.utils.env import get_env
from mlflow.utils.file_utils import (is_directory, list_subdirs, mkdir, exists, write_yaml,
                                     read_yaml, find, read_file_lines, read_file, build_path,
                                     write_to, append_to, make_containing_dirs, mv, get_parent_dir,
                                     list_all)
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_PARENT_RUN_ID

_TRACKING_DIR_ENV_VAR = "MLFLOW_TRACKING_DIR"


def _default_root_dir():
    return get_env(_TRACKING_DIR_ENV_VAR) or os.path.abspath(DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH)


def _make_persisted_run_info_dict(run_info):
    # 'tags' was moved from RunInfo to RunData, so we must keep storing it in the meta.yaml for
    # old mlflow versions to read
    run_info_dict = dict(run_info)
    run_info_dict['tags'] = []
    return run_info_dict


def _read_persisted_run_info_dict(run_info_dict):
    dict_copy = run_info_dict.copy()
    if 'lifecycle_stage' not in dict_copy:
        dict_copy['lifecycle_stage'] = LifecycleStage.ACTIVE
    return RunInfo.from_dictionary(dict_copy)


class FileStore(AbstractStore):
    TRASH_FOLDER_NAME = ".trash"
    ARTIFACTS_FOLDER_NAME = "artifacts"
    METRICS_FOLDER_NAME = "metrics"
    PARAMS_FOLDER_NAME = "params"
    TAGS_FOLDER_NAME = "tags"
    META_DATA_FILE_NAME = "meta.yaml"

    def __init__(self, root_directory=None, artifact_root_uri=None):
        """
        Create a new FileStore with the given root directory and a given default artifact root URI.
        """
        super(FileStore, self).__init__()
        self.root_directory = root_directory or _default_root_dir()
        self.artifact_root_uri = artifact_root_uri or self.root_directory
        self.trash_folder = build_path(self.root_directory, FileStore.TRASH_FOLDER_NAME)
        # Create root directory if needed
        if not exists(self.root_directory):
            mkdir(self.root_directory)
            self._create_experiment_with_id(name=Experiment.DEFAULT_EXPERIMENT_NAME,
                                            experiment_id=Experiment.DEFAULT_EXPERIMENT_ID,
                                            artifact_uri=None)
        # Create trash folder if needed
        if not exists(self.trash_folder):
            mkdir(self.trash_folder)

    def _check_root_dir(self):
        """
        Run checks before running directory operations.
        """
        if not exists(self.root_directory):
            raise Exception("'%s' does not exist." % self.root_directory)
        if not is_directory(self.root_directory):
            raise Exception("'%s' is not a directory." % self.root_directory)

    def _get_experiment_path(self, experiment_id, view_type=ViewType.ALL, assert_exists=False):
        parents = []
        if view_type == ViewType.ACTIVE_ONLY or view_type == ViewType.ALL:
            parents.append(self.root_directory)
        if view_type == ViewType.DELETED_ONLY or view_type == ViewType.ALL:
            parents.append(self.trash_folder)
        for parent in parents:
            exp_list = find(parent, str(experiment_id), full_path=True)
            if len(exp_list) > 0:
                return exp_list[0]
        if assert_exists:
            raise MlflowException('Experiment {} does not exist.'.format(experiment_id),
                                  databricks_pb2.RESOURCE_DOES_NOT_EXIST)
        return None

    def _get_run_dir(self, experiment_id, run_uuid):
        _validate_run_id(run_uuid)
        if not self._has_experiment(experiment_id):
            return None
        return build_path(self._get_experiment_path(experiment_id, assert_exists=True), run_uuid)

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

    def _get_tag_path(self, experiment_id, run_uuid, tag_name):
        _validate_run_id(run_uuid)
        _validate_tag_name(tag_name)
        return build_path(self._get_run_dir(experiment_id, run_uuid), FileStore.TAGS_FOLDER_NAME,
                          tag_name)

    def _get_artifact_dir(self, experiment_id, run_uuid):
        _validate_run_id(run_uuid)
        artifacts_dir = build_path(self.get_experiment(experiment_id).artifact_location,
                                   run_uuid,
                                   FileStore.ARTIFACTS_FOLDER_NAME)
        return artifacts_dir

    def _get_active_experiments(self, full_path=False):
        exp_list = list_subdirs(self.root_directory, full_path)
        return [exp for exp in exp_list if not exp.endswith(FileStore.TRASH_FOLDER_NAME)]

    def _get_deleted_experiments(self, full_path=False):
        return list_subdirs(self.trash_folder, full_path)

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        self._check_root_dir()
        rsl = []
        if view_type == ViewType.ACTIVE_ONLY or view_type == ViewType.ALL:
            rsl += self._get_active_experiments(full_path=False)
        if view_type == ViewType.DELETED_ONLY or view_type == ViewType.ALL:
            rsl += self._get_deleted_experiments(full_path=False)
        experiments = []
        for exp_id in rsl:
            try:
                # trap and warn known issues, will raise unexpected exceptions to caller
                experiment = self._get_experiment(exp_id, view_type)
                if experiment:
                    experiments.append(experiment)
            except MissingConfigException as rnfe:
                # Trap malformed experiments and log warnings.
                logging.warning("Malformed experiment '%s'. Detailed error %s",
                                str(exp_id), str(rnfe), exc_info=True)
        return experiments

    def _create_experiment_with_id(self, name, experiment_id, artifact_uri):
        self._check_root_dir()
        meta_dir = mkdir(self.root_directory, str(experiment_id))
        artifact_uri = artifact_uri or build_path(self.artifact_root_uri, str(experiment_id))
        experiment = Experiment(experiment_id, name, artifact_uri, LifecycleStage.ACTIVE)
        write_yaml(meta_dir, FileStore.META_DATA_FILE_NAME, dict(experiment))
        return experiment_id

    def create_experiment(self, name, artifact_location=None):
        self._check_root_dir()
        if name is None or name == "":
            raise MlflowException("Invalid experiment name '%s'" % name,
                                  databricks_pb2.INVALID_PARAMETER_VALUE)
        experiment = self.get_experiment_by_name(name)
        if experiment is not None:
            if experiment.lifecycle_stage == LifecycleStage.DELETED:
                raise MlflowException(
                    "Experiment '%s' already exists in deleted state. "
                    "You can restore the experiment, or permanently delete the experiment "
                    "from the .trash folder (under tracking server's root folder) before "
                    "creating a new one with the same name." % experiment.name,
                    databricks_pb2.RESOURCE_ALREADY_EXISTS)
            else:
                raise MlflowException("Experiment '%s' already exists." % experiment.name,
                                      databricks_pb2.RESOURCE_ALREADY_EXISTS)
        # Get all existing experiments and find the one with largest ID.
        # len(list_all(..)) would not work when experiments are deleted.
        experiments_ids = [e.experiment_id for e in self.list_experiments(ViewType.ALL)]
        experiment_id = max(experiments_ids) + 1 if experiments_ids else 0
        return self._create_experiment_with_id(name, experiment_id, artifact_location)

    def _has_experiment(self, experiment_id):
        return self._get_experiment_path(experiment_id) is not None

    def _get_experiment(self, experiment_id, view_type=ViewType.ALL):
        self._check_root_dir()
        _validate_experiment_id(experiment_id)
        experiment_dir = self._get_experiment_path(experiment_id, view_type)
        if experiment_dir is None:
            raise MlflowException("Could not find experiment with ID %s" % experiment_id,
                                  databricks_pb2.RESOURCE_DOES_NOT_EXIST)
        meta = read_yaml(experiment_dir, FileStore.META_DATA_FILE_NAME)
        if experiment_dir.startswith(self.trash_folder):
            meta['lifecycle_stage'] = LifecycleStage.DELETED
        else:
            meta['lifecycle_stage'] = LifecycleStage.ACTIVE
        experiment = Experiment.from_dictionary(meta)
        if int(experiment_id) != experiment.experiment_id:
            logging.warning("Experiment ID mismatch for exp %s. ID recorded as '%s' in meta data. "
                            "Experiment will be ignored.",
                            str(experiment_id), str(experiment.experiment_id), exc_info=True)
            return None
        return experiment

    def get_experiment(self, experiment_id):
        """
        Fetch the experiment.
        Note: This API will search for active as well as deleted experiments.

        :param experiment_id: Integer id for the experiment
        :return: A single Experiment object if it exists, otherwise raises an Exception.
        """
        experiment = self._get_experiment(experiment_id)
        if experiment is None:
            raise MlflowException("Experiment '%s' does not exist." % experiment_id,
                                  databricks_pb2.RESOURCE_DOES_NOT_EXIST)
        return experiment

    def delete_experiment(self, experiment_id):
        experiment_dir = self._get_experiment_path(experiment_id, ViewType.ACTIVE_ONLY)
        if experiment_dir is None:
            raise MlflowException("Could not find experiment with ID %s" % experiment_id,
                                  databricks_pb2.RESOURCE_DOES_NOT_EXIST)
        mv(experiment_dir, self.trash_folder)

    def restore_experiment(self, experiment_id):
        experiment_dir = self._get_experiment_path(experiment_id, ViewType.DELETED_ONLY)
        if experiment_dir is None:
            raise MlflowException("Could not find deleted experiment with ID %d" % experiment_id,
                                  databricks_pb2.RESOURCE_DOES_NOT_EXIST)
        conflict_experiment = self._get_experiment_path(experiment_id, ViewType.ACTIVE_ONLY)
        if conflict_experiment is not None:
            raise MlflowException(
                    "Cannot restore eperiment with ID %d. "
                    "An experiment with same ID already exists." % experiment_id,
                    databricks_pb2.RESOURCE_ALREADY_EXISTS)
        mv(experiment_dir, self.root_directory)

    def rename_experiment(self, experiment_id, new_name):
        meta_dir = os.path.join(self.root_directory, str(experiment_id))
        # if experiment is malformed, will raise error
        experiment = self._get_experiment(experiment_id)
        if experiment is None:
            raise MlflowException("Experiment '%s' does not exist." % experiment_id,
                                  databricks_pb2.RESOURCE_DOES_NOT_EXIST)
        experiment._set_name(new_name)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise Exception("Cannot rename experiment in non-active lifecycle stage."
                            " Current stage: %s" % experiment.lifecycle_stage)
        write_yaml(meta_dir, FileStore.META_DATA_FILE_NAME, dict(experiment), overwrite=True)

    def delete_run(self, run_id):
        run_info = self._get_run_info(run_id)
        if run_info is None:
            raise MlflowException("Run '%s' metadata is in invalid state." % run_id,
                                  databricks_pb2.INVALID_STATE)
        check_run_is_active(run_info)
        new_info = run_info._copy_with_overrides(lifecycle_stage=LifecycleStage.DELETED)
        self._overwrite_run_info(new_info)

    def restore_run(self, run_id):
        run_info = self._get_run_info(run_id)
        if run_info is None:
            raise MlflowException("Run '%s' metadata is in invalid state." % run_id,
                                  databricks_pb2.INVALID_STATE)
        check_run_is_deleted(run_info)
        new_info = run_info._copy_with_overrides(lifecycle_stage=LifecycleStage.ACTIVE)
        self._overwrite_run_info(new_info)

    def _find_experiment_folder(self, run_path):
        """
        Given a run path, return the parent directory for its experiment.
        """
        parent = get_parent_dir(run_path)
        if os.path.basename(parent) == FileStore.TRASH_FOLDER_NAME:
            return get_parent_dir(parent)
        return parent

    def _find_run_root(self, run_uuid):
        _validate_run_id(run_uuid)
        self._check_root_dir()
        all_experiments = self._get_active_experiments(True) + self._get_deleted_experiments(True)
        for experiment_dir in all_experiments:
            runs = find(experiment_dir, run_uuid, full_path=True)
            if len(runs) == 0:
                continue
            return os.path.basename(os.path.abspath(experiment_dir)), runs[0]
        return None, None

    def update_run_info(self, run_uuid, run_status, end_time):
        _validate_run_id(run_uuid)
        run_info = self.get_run(run_uuid).info
        check_run_is_active(run_info)
        new_info = run_info._copy_with_overrides(run_status, end_time)
        self._overwrite_run_info(new_info)
        return new_info

    def create_run(self, experiment_id, user_id, run_name, source_type,
                   source_name, entry_point_name, start_time, source_version, tags, parent_run_id):
        """
        Creates a run with the specified attributes.
        """
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise MlflowException(
                    "Could not create run under experiment with ID %s - no such experiment "
                    "exists." % experiment_id,
                    databricks_pb2.RESOURCE_DOES_NOT_EXIST)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                    "Could not create run under non-active experiment with ID "
                    "%s." % experiment_id,
                    databricks_pb2.INVALID_STATE)
        run_uuid = uuid.uuid4().hex
        artifact_uri = self._get_artifact_dir(experiment_id, run_uuid)
        run_info = RunInfo(run_uuid=run_uuid, experiment_id=experiment_id,
                           name="",
                           artifact_uri=artifact_uri, source_type=source_type,
                           source_name=source_name,
                           entry_point_name=entry_point_name, user_id=user_id,
                           status=RunStatus.RUNNING, start_time=start_time, end_time=None,
                           source_version=source_version, lifecycle_stage=LifecycleStage.ACTIVE)
        # Persist run metadata and create directories for logging metrics, parameters, artifacts
        run_dir = self._get_run_dir(run_info.experiment_id, run_info.run_uuid)
        mkdir(run_dir)
        write_yaml(run_dir, FileStore.META_DATA_FILE_NAME, _make_persisted_run_info_dict(run_info))
        mkdir(run_dir, FileStore.METRICS_FOLDER_NAME)
        mkdir(run_dir, FileStore.PARAMS_FOLDER_NAME)
        mkdir(run_dir, FileStore.ARTIFACTS_FOLDER_NAME)
        for tag in tags:
            self.set_tag(run_uuid, tag)
        if parent_run_id:
            self.set_tag(run_uuid, RunTag(key=MLFLOW_PARENT_RUN_ID, value=parent_run_id))
        if run_name:
            self.set_tag(run_uuid, RunTag(key=MLFLOW_RUN_NAME, value=run_name))
        return Run(run_info=run_info, run_data=None)

    def _make_experiment_dict(self, experiment):
        # Don't persist lifecycle_stage since it's inferred from the ".trash" folder.
        experiment_dict = dict(experiment)
        del experiment_dict['lifecycle_stage']
        return experiment_dict

    def get_run(self, run_uuid):
        """
        Note: Will get both active and deleted runs.
        """
        _validate_run_id(run_uuid)
        run_info = self._get_run_info(run_uuid)
        if run_info is None:
            raise MlflowException("Run '%s' metadata is in invalid state." % run_uuid,
                                  databricks_pb2.INVALID_STATE)
        metrics = self.get_all_metrics(run_uuid)
        params = self.get_all_params(run_uuid)
        tags = self.get_all_tags(run_uuid)
        return Run(run_info, RunData(metrics, params, tags))

    def _get_run_info(self, run_uuid):
        """
        Note: Will get both active and deleted runs.
        """
        exp_id, run_dir = self._find_run_root(run_uuid)
        if run_dir is None:
            raise MlflowException("Run '%s' not found" % run_uuid,
                                  databricks_pb2.RESOURCE_DOES_NOT_EXIST)

        meta = read_yaml(run_dir, FileStore.META_DATA_FILE_NAME)
        run_info = _read_persisted_run_info_dict(meta)
        if str(run_info.experiment_id) != str(exp_id):
            logging.warning("Wrong experiment ID (%s) recorded for run '%s'. It should be %s. "
                            "Run will be ignored.", str(run_info.experiment_id),
                            str(run_info.run_uuid), str(exp_id), exc_info=True)
            return None
        return run_info

    def _get_run_files(self, run_uuid, resource_type):
        _validate_run_id(run_uuid)
        run_info = self._get_run_info(run_uuid)
        if run_info is None:
            raise MlflowException("Run '%s' metadata is in invalid state." % run_uuid,
                                  databricks_pb2.INVALID_STATE)
        if resource_type == "metric":
            subfolder_name = FileStore.METRICS_FOLDER_NAME
        elif resource_type == "param":
            subfolder_name = FileStore.PARAMS_FOLDER_NAME
        elif resource_type == "tag":
            subfolder_name = FileStore.TAGS_FOLDER_NAME
        else:
            raise Exception("Looking for unknown resource under run.")
        _, run_dir = self._find_run_root(run_uuid)
        # run_dir exists since run validity has been confirmed above.
        source_dirs = find(run_dir, subfolder_name, full_path=True)
        if len(source_dirs) == 0:
            return run_dir, []
        file_names = []
        for root, _, files in os.walk(source_dirs[0]):
            for name in files:
                abspath = os.path.join(root, name)
                file_names.append(os.path.relpath(abspath, source_dirs[0]))
        return source_dirs[0], file_names

    @staticmethod
    def _get_metric_from_file(parent_path, metric_name):
        _validate_metric_name(metric_name)
        metric_data = read_file_lines(parent_path, metric_name)
        if len(metric_data) == 0:
            raise Exception("Metric '%s' is malformed. No data found." % metric_name)
        last_line = metric_data[-1]
        timestamp, val = last_line.strip().split(" ")
        return Metric(metric_name, float(val), int(timestamp))

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
            raise MlflowException("Metric '%s' not found under run '%s'" % (metric_key, run_uuid),
                                  databricks_pb2.RESOURCE_DOES_NOT_EXIST)
        metric_data = read_file_lines(parent_path, metric_key)
        rsl = []
        for pair in metric_data:
            ts, val = pair.strip().split(" ")
            rsl.append(Metric(metric_key, float(val), int(ts)))
        return rsl

    @staticmethod
    def _get_param_from_file(parent_path, param_name):
        _validate_param_name(param_name)
        param_data = read_file_lines(parent_path, param_name)
        if len(param_data) > 1:
            raise Exception("Unexpected data for param '%s'. Param recorded more than once"
                            % param_name)
        # The only cause for param_data's length to be zero is the param's
        # value is an empty string
        value = '' if len(param_data) == 0 else str(param_data[0].strip())
        return Param(param_name, value)

    @staticmethod
    def _get_tag_from_file(parent_path, tag_name):
        _validate_tag_name(tag_name)
        tag_data = read_file(parent_path, tag_name)
        return RunTag(tag_name, tag_data)

    def get_all_params(self, run_uuid):
        parent_path, param_files = self._get_run_files(run_uuid, "param")
        params = []
        for param_file in param_files:
            params.append(self._get_param_from_file(parent_path, param_file))
        return params

    def get_all_tags(self, run_uuid):
        parent_path, tag_files = self._get_run_files(run_uuid, "tag")
        tags = []
        for tag_file in tag_files:
            tags.append(self._get_tag_from_file(parent_path, tag_file))
        return tags

    def _list_run_infos(self, experiment_id, view_type):
        self._check_root_dir()
        if not self._has_experiment(experiment_id):
            return []
        experiment_dir = self._get_experiment_path(experiment_id, assert_exists=True)
        run_uuids = list_all(experiment_dir, os.path.isdir, full_path=False)
        run_infos = []
        for r_id in run_uuids:
            try:
                # trap and warn known issues, will raise unexpected exceptions to caller
                run_info = self._get_run_info(r_id)
                if run_info is None:
                    continue
                if LifecycleStage.matches_view_type(view_type, run_info.lifecycle_stage):
                    run_infos.append(run_info)
            except MissingConfigException as rnfe:
                # trap malformed run exception and log warning
                logging.warning("Malformed run '%s'. Detailed error %s", r_id, str(rnfe),
                                exc_info=True)
        return run_infos

    def search_runs(self, experiment_ids, search_filter, run_view_type):
        runs = []
        for experiment_id in experiment_ids:
            run_infos = self._list_run_infos(experiment_id, run_view_type)
            runs.extend(self.get_run(r.run_uuid) for r in run_infos)
        return [run for run in runs if not search_filter or search_filter.filter(run)]

    def log_metric(self, run_uuid, metric):
        _validate_run_id(run_uuid)
        _validate_metric_name(metric.key)
        run = self.get_run(run_uuid)
        check_run_is_active(run.info)
        metric_path = self._get_metric_path(run.info.experiment_id, run_uuid, metric.key)
        make_containing_dirs(metric_path)
        append_to(metric_path, "%s %s\n" % (metric.timestamp, metric.value))

    def _writeable_value(self, tag_value):
        if tag_value is None:
            return ""
        elif isinstance(tag_value, six.string_types):
            return tag_value
        else:
            return "%s" % tag_value

    def log_param(self, run_uuid, param):
        _validate_run_id(run_uuid)
        _validate_param_name(param.key)
        run = self.get_run(run_uuid)
        check_run_is_active(run.info)
        param_path = self._get_param_path(run.info.experiment_id, run_uuid, param.key)
        make_containing_dirs(param_path)
        write_to(param_path, self._writeable_value(param.value))

    def set_tag(self, run_uuid, tag):
        _validate_run_id(run_uuid)
        _validate_tag_name(tag.key)
        run = self.get_run(run_uuid)
        check_run_is_active(run.info)
        tag_path = self._get_tag_path(run.info.experiment_id, run_uuid, tag.key)
        make_containing_dirs(tag_path)
        # Don't add trailing newline
        write_to(tag_path, self._writeable_value(tag.value))

    def _overwrite_run_info(self, run_info):
        run_dir = self._get_run_dir(run_info.experiment_id, run_info.run_uuid)
        run_info_dict = _make_persisted_run_info_dict(run_info)
        write_yaml(run_dir, FileStore.META_DATA_FILE_NAME, run_info_dict, overwrite=True)

    def log_batch(self, run_id, metrics, params, tags):
        _validate_run_id(run_id)
        run = self.get_run(run_id)
        check_run_is_active(run.info)
        try:
            for param in params:
                self.log_param(run_id, param)
            for metric in metrics:
                self.log_metric(run_id, metric)
            for tag in tags:
                self.set_tag(run_id, tag)
        except Exception as e:
            raise MlflowException(e, INTERNAL_ERROR)
