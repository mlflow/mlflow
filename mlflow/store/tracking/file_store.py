import json
import logging
import time
import os
import sys
import shutil

import uuid

from mlflow.entities import (
    Experiment,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunStatus,
    RunTag,
    ViewType,
    SourceType,
    ExperimentTag,
)
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.run_info import check_run_is_active
from mlflow.exceptions import MlflowException, MissingConfigException
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    RESOURCE_DOES_NOT_EXIST,
    INVALID_PARAMETER_VALUE,
)
from mlflow.store.model_registry.file_store import FileStore as ModelRegistryFileStore
from mlflow.store.tracking import (
    DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH,
    SEARCH_MAX_RESULTS_DEFAULT,
    SEARCH_MAX_RESULTS_THRESHOLD,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.name_utils import _generate_random_name, _generate_unique_integer_id
from mlflow.utils.validation import (
    _validate_metric,
    _validate_metric_name,
    _validate_param_name,
    _validate_param,
    _validate_run_id,
    _validate_tag_name,
    _validate_experiment_id,
    _validate_batch_log_limits,
    _validate_batch_log_data,
    _validate_param_keys_unique,
    _validate_experiment_name,
)
from mlflow.utils.env import get_env
from mlflow.utils.file_utils import (
    is_directory,
    list_subdirs,
    mkdir,
    exists,
    write_yaml,
    overwrite_yaml,
    read_yaml,
    find,
    read_file_lines,
    read_file,
    write_to,
    append_to,
    make_containing_dirs,
    mv,
    get_parent_dir,
    list_all,
    local_file_uri_to_path,
    path_to_local_file_uri,
)
from mlflow.utils.search_utils import SearchUtils, SearchExperimentsUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time_utils import get_current_time_millis
from mlflow.utils.uri import (
    append_to_uri_path,
    resolve_uri_if_local,
)
from mlflow.utils.mlflow_tags import MLFLOW_LOGGED_MODELS, MLFLOW_RUN_NAME, _get_run_name_from_tags

_TRACKING_DIR_ENV_VAR = "MLFLOW_TRACKING_DIR"


def _default_root_dir():
    return get_env(_TRACKING_DIR_ENV_VAR) or os.path.abspath(DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH)


def _read_persisted_experiment_dict(experiment_dict):
    dict_copy = experiment_dict.copy()

    # 'experiment_id' was changed from int to string, so we must cast to string
    # when reading legacy experiments
    if isinstance(dict_copy["experiment_id"], int):
        dict_copy["experiment_id"] = str(dict_copy["experiment_id"])
    return Experiment.from_dictionary(dict_copy)


def _make_persisted_run_info_dict(run_info):
    # 'tags' was moved from RunInfo to RunData, so we must keep storing it in the meta.yaml for
    # old mlflow versions to read
    run_info_dict = dict(run_info)
    run_info_dict["tags"] = []
    if "status" in run_info_dict:
        # 'status' is stored as an integer enum in meta file, but RunInfo.status field is a string.
        # Convert from string to enum/int before storing.
        run_info_dict["status"] = RunStatus.from_string(run_info.status)
    else:
        run_info_dict["status"] = RunStatus.RUNNING
    run_info_dict["source_type"] = SourceType.LOCAL
    run_info_dict["source_name"] = ""
    run_info_dict["entry_point_name"] = ""
    run_info_dict["source_version"] = ""
    return run_info_dict


def _read_persisted_run_info_dict(run_info_dict):
    dict_copy = run_info_dict.copy()
    if "lifecycle_stage" not in dict_copy:
        dict_copy["lifecycle_stage"] = LifecycleStage.ACTIVE
    # 'status' is stored as an integer enum in meta file, but RunInfo.status field is a string.
    # converting to string before hydrating RunInfo.
    # If 'status' value not recorded in files, mark it as 'RUNNING' (default)
    dict_copy["status"] = RunStatus.to_string(run_info_dict.get("status", RunStatus.RUNNING))

    # 'experiment_id' was changed from int to string, so we must cast to string
    # when reading legacy run_infos
    if isinstance(dict_copy["experiment_id"], int):
        dict_copy["experiment_id"] = str(dict_copy["experiment_id"])
    return RunInfo.from_dictionary(dict_copy)


class FileStore(AbstractStore):
    TRASH_FOLDER_NAME = ".trash"
    ARTIFACTS_FOLDER_NAME = "artifacts"
    METRICS_FOLDER_NAME = "metrics"
    PARAMS_FOLDER_NAME = "params"
    TAGS_FOLDER_NAME = "tags"
    EXPERIMENT_TAGS_FOLDER_NAME = "tags"
    RESERVED_EXPERIMENT_FOLDERS = [EXPERIMENT_TAGS_FOLDER_NAME]
    META_DATA_FILE_NAME = "meta.yaml"
    DEFAULT_EXPERIMENT_ID = "0"

    def __init__(self, root_directory=None, artifact_root_uri=None):
        """
        Create a new FileStore with the given root directory and a given default artifact root URI.
        """
        super().__init__()
        self.root_directory = local_file_uri_to_path(root_directory or _default_root_dir())
        if not artifact_root_uri:
            self.artifact_root_uri = path_to_local_file_uri(self.root_directory)
        else:
            self.artifact_root_uri = resolve_uri_if_local(artifact_root_uri)
        self.trash_folder = os.path.join(self.root_directory, FileStore.TRASH_FOLDER_NAME)
        # Create root directory if needed
        if not exists(self.root_directory):
            self._create_default_experiment()
        # Create trash folder if needed
        if not exists(self.trash_folder):
            mkdir(self.trash_folder)

    def _create_default_experiment(self):
        mkdir(self.root_directory)
        self._create_experiment_with_id(
            name=Experiment.DEFAULT_EXPERIMENT_NAME,
            experiment_id=FileStore.DEFAULT_EXPERIMENT_ID,
            artifact_uri=None,
            tags=None,
        )

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
            exp_list = find(parent, experiment_id, full_path=True)
            if len(exp_list) > 0:
                return exp_list[0]
        if assert_exists:
            raise MlflowException(
                f"Experiment {experiment_id} does not exist.",
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        return None

    def _get_run_dir(self, experiment_id, run_uuid):
        _validate_run_id(run_uuid)
        if not self._has_experiment(experiment_id):
            return None
        return os.path.join(self._get_experiment_path(experiment_id, assert_exists=True), run_uuid)

    def _get_metric_path(self, experiment_id, run_uuid, metric_key):
        _validate_run_id(run_uuid)
        _validate_metric_name(metric_key)
        return os.path.join(
            self._get_run_dir(experiment_id, run_uuid), FileStore.METRICS_FOLDER_NAME, metric_key
        )

    def _get_param_path(self, experiment_id, run_uuid, param_name):
        _validate_run_id(run_uuid)
        _validate_param_name(param_name)
        return os.path.join(
            self._get_run_dir(experiment_id, run_uuid), FileStore.PARAMS_FOLDER_NAME, param_name
        )

    def _get_experiment_tag_path(self, experiment_id, tag_name):
        _validate_experiment_id(experiment_id)
        _validate_param_name(tag_name)
        if not self._has_experiment(experiment_id):
            return None
        return os.path.join(
            self._get_experiment_path(experiment_id, assert_exists=True),
            FileStore.TAGS_FOLDER_NAME,
            tag_name,
        )

    def _get_tag_path(self, experiment_id, run_uuid, tag_name):
        _validate_run_id(run_uuid)
        _validate_tag_name(tag_name)
        return os.path.join(
            self._get_run_dir(experiment_id, run_uuid), FileStore.TAGS_FOLDER_NAME, tag_name
        )

    def _get_artifact_dir(self, experiment_id, run_uuid):
        _validate_run_id(run_uuid)
        return append_to_uri_path(
            self.get_experiment(experiment_id).artifact_location,
            run_uuid,
            FileStore.ARTIFACTS_FOLDER_NAME,
        )

    def _get_active_experiments(self, full_path=False):
        exp_list = list_subdirs(self.root_directory, full_path)
        return [
            exp
            for exp in exp_list
            if not exp.endswith(FileStore.TRASH_FOLDER_NAME)
            and exp != ModelRegistryFileStore.MODELS_FOLDER_NAME
        ]

    def _get_deleted_experiments(self, full_path=False):
        return list_subdirs(self.trash_folder, full_path)

    def search_experiments(
        self,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        filter_string=None,
        order_by=None,
        page_token=None,
    ):
        if not isinstance(max_results, int) or max_results < 1:
            raise MlflowException(
                "Invalid value for max_results. It must be a positive integer,"
                f" but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )
        if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                f"Invalid value for max_results. It must be at most {SEARCH_MAX_RESULTS_THRESHOLD},"
                f" but got {max_results}",
                INVALID_PARAMETER_VALUE,
            )

        self._check_root_dir()
        experiment_ids = []
        if view_type == ViewType.ACTIVE_ONLY or view_type == ViewType.ALL:
            experiment_ids += self._get_active_experiments(full_path=False)
        if view_type == ViewType.DELETED_ONLY or view_type == ViewType.ALL:
            experiment_ids += self._get_deleted_experiments(full_path=False)

        experiments = []
        for exp_id in experiment_ids:
            try:
                # trap and warn known issues, will raise unexpected exceptions to caller
                exp = self._get_experiment(exp_id, view_type)
                if exp is not None:
                    experiments.append(exp)
            except MissingConfigException as e:
                logging.warning(
                    f"Malformed experiment '{exp_id}'. Detailed error {e}", exc_info=True
                )
        filtered = SearchExperimentsUtils.filter(experiments, filter_string)
        sorted_experiments = SearchExperimentsUtils.sort(
            filtered, order_by or ["creation_time DESC", "experiment_id ASC"]
        )
        experiments, next_page_token = SearchUtils.paginate(
            sorted_experiments, page_token, max_results
        )
        return PagedList(experiments, next_page_token)

    def get_experiment_by_name(self, experiment_name):
        def pagination_wrapper_func(number_to_get, next_page_token):
            return self.search_experiments(
                view_type=ViewType.ACTIVE_ONLY,
                max_results=number_to_get,
                filter_string=f"name = '{experiment_name}'",
                page_token=next_page_token,
            )

        experiments = get_results_from_paginated_fn(
            paginated_fn=pagination_wrapper_func,
            max_results_per_page=SEARCH_MAX_RESULTS_THRESHOLD,
            max_results=None,
        )
        return experiments[0] if len(experiments) > 0 else None

    def _create_experiment_with_id(self, name, experiment_id, artifact_uri, tags):
        if not artifact_uri:
            resolved_artifact_uri = append_to_uri_path(self.artifact_root_uri, str(experiment_id))
        else:
            resolved_artifact_uri = resolve_uri_if_local(artifact_uri)
        meta_dir = mkdir(self.root_directory, str(experiment_id))
        creation_time = get_current_time_millis()
        experiment = Experiment(
            experiment_id,
            name,
            resolved_artifact_uri,
            LifecycleStage.ACTIVE,
            creation_time=creation_time,
            last_update_time=creation_time,
        )
        experiment_dict = dict(experiment)
        # tags are added to the file system and are not written to this dict on write
        # As such, we should not include them in the meta file.
        del experiment_dict["tags"]
        write_yaml(meta_dir, FileStore.META_DATA_FILE_NAME, experiment_dict)
        if tags is not None:
            for tag in tags:
                self.set_experiment_tag(experiment_id, tag)
        return experiment_id

    def _validate_experiment_does_not_exist(self, name):
        experiment = self.get_experiment_by_name(name)
        if experiment is not None:
            if experiment.lifecycle_stage == LifecycleStage.DELETED:
                raise MlflowException(
                    "Experiment '%s' already exists in deleted state. "
                    "You can restore the experiment, or permanently delete the experiment "
                    "from the .trash folder (under tracking server's root folder) in order to "
                    "use this experiment name again." % experiment.name,
                    databricks_pb2.RESOURCE_ALREADY_EXISTS,
                )
            else:
                raise MlflowException(
                    "Experiment '%s' already exists." % experiment.name,
                    databricks_pb2.RESOURCE_ALREADY_EXISTS,
                )

    def create_experiment(self, name, artifact_location=None, tags=None):
        self._check_root_dir()
        _validate_experiment_name(name)
        self._validate_experiment_does_not_exist(name)
        experiment_id = _generate_unique_integer_id()
        return self._create_experiment_with_id(name, str(experiment_id), artifact_location, tags)

    def _has_experiment(self, experiment_id):
        return self._get_experiment_path(experiment_id) is not None

    def _get_experiment(self, experiment_id, view_type=ViewType.ALL):
        self._check_root_dir()
        _validate_experiment_id(experiment_id)
        experiment_dir = self._get_experiment_path(experiment_id, view_type)
        if experiment_dir is None:
            raise MlflowException(
                "Could not find experiment with ID %s" % experiment_id,
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        meta = FileStore._read_yaml(experiment_dir, FileStore.META_DATA_FILE_NAME)
        if experiment_dir.startswith(self.trash_folder):
            meta["lifecycle_stage"] = LifecycleStage.DELETED
        else:
            meta["lifecycle_stage"] = LifecycleStage.ACTIVE
        meta["tags"] = self.get_all_experiment_tags(experiment_id)
        experiment = _read_persisted_experiment_dict(meta)
        if experiment_id != experiment.experiment_id:
            logging.warning(
                "Experiment ID mismatch for exp %s. ID recorded as '%s' in meta data. "
                "Experiment will be ignored.",
                experiment_id,
                experiment.experiment_id,
                exc_info=True,
            )
            return None
        return experiment

    def get_experiment(self, experiment_id):
        """
        Fetch the experiment.
        Note: This API will search for active as well as deleted experiments.

        :param experiment_id: Integer id for the experiment
        :return: A single Experiment object if it exists, otherwise raises an Exception.
        """
        experiment_id = FileStore.DEFAULT_EXPERIMENT_ID if experiment_id is None else experiment_id
        experiment = self._get_experiment(experiment_id)
        if experiment is None:
            raise MlflowException(
                "Experiment '%s' does not exist." % experiment_id,
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        return experiment

    def delete_experiment(self, experiment_id):
        if str(experiment_id) == str(FileStore.DEFAULT_EXPERIMENT_ID):
            raise MlflowException(
                "Cannot delete the default experiment "
                f"'{FileStore.DEFAULT_EXPERIMENT_ID}'. This is an internally "
                f"reserved experiment."
            )
        experiment_dir = self._get_experiment_path(experiment_id, ViewType.ACTIVE_ONLY)
        if experiment_dir is None:
            raise MlflowException(
                "Could not find experiment with ID %s" % experiment_id,
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        experiment = self._get_experiment(experiment_id)
        experiment._set_last_update_time(get_current_time_millis())
        meta_dir = os.path.join(self.root_directory, experiment_id)
        overwrite_yaml(
            root=meta_dir,
            file_name=FileStore.META_DATA_FILE_NAME,
            data=dict(experiment),
        )
        mv(experiment_dir, self.trash_folder)

    def _hard_delete_experiment(self, experiment_id):
        """
        Permanently delete an experiment.
        This is used by the ``mlflow gc`` command line and is not intended to be used elsewhere.
        """
        experiment_dir = self._get_experiment_path(experiment_id, ViewType.DELETED_ONLY)
        shutil.rmtree(experiment_dir)

    def restore_experiment(self, experiment_id):
        experiment_dir = self._get_experiment_path(experiment_id, ViewType.DELETED_ONLY)
        if experiment_dir is None:
            raise MlflowException(
                "Could not find deleted experiment with ID %d" % experiment_id,
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        conflict_experiment = self._get_experiment_path(experiment_id, ViewType.ACTIVE_ONLY)
        if conflict_experiment is not None:
            raise MlflowException(
                "Cannot restore experiment with ID %d. "
                "An experiment with same ID already exists." % experiment_id,
                databricks_pb2.RESOURCE_ALREADY_EXISTS,
            )
        mv(experiment_dir, self.root_directory)
        experiment = self._get_experiment(experiment_id)
        meta_dir = os.path.join(self.root_directory, experiment_id)
        experiment._set_last_update_time(get_current_time_millis())
        overwrite_yaml(
            root=meta_dir,
            file_name=FileStore.META_DATA_FILE_NAME,
            data=dict(experiment),
        )

    def rename_experiment(self, experiment_id, new_name):
        _validate_experiment_name(new_name)
        meta_dir = os.path.join(self.root_directory, experiment_id)
        # if experiment is malformed, will raise error
        experiment = self._get_experiment(experiment_id)
        if experiment is None:
            raise MlflowException(
                "Experiment '%s' does not exist." % experiment_id,
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        self._validate_experiment_does_not_exist(new_name)
        experiment._set_name(new_name)
        experiment._set_last_update_time(get_current_time_millis())
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise Exception(
                "Cannot rename experiment in non-active lifecycle stage."
                " Current stage: %s" % experiment.lifecycle_stage
            )
        overwrite_yaml(
            root=meta_dir,
            file_name=FileStore.META_DATA_FILE_NAME,
            data=dict(experiment),
        )

    def delete_run(self, run_id):
        run_info = self._get_run_info(run_id)
        if run_info is None:
            raise MlflowException(
                "Run '%s' metadata is in invalid state." % run_id, databricks_pb2.INVALID_STATE
            )
        new_info = run_info._copy_with_overrides(lifecycle_stage=LifecycleStage.DELETED)
        self._overwrite_run_info(new_info, deleted_time=get_current_time_millis())

    def _hard_delete_run(self, run_id):
        """
        Permanently delete a run (metadata and metrics, tags, parameters).
        This is used by the ``mlflow gc`` command line and is not intended to be used elsewhere.
        """
        _, run_dir = self._find_run_root(run_id)
        shutil.rmtree(run_dir)

    def _get_deleted_runs(self, older_than=0):
        """
        Get all deleted run ids.
        Args:
            older_than: get runs that is older than this variable in number of milliseconds.
                        defaults to 0 ms to get all deleted runs.
        """
        current_time = get_current_time_millis()
        experiment_ids = self._get_active_experiments() + self._get_deleted_experiments()
        deleted_runs = self.search_runs(
            experiment_ids=experiment_ids, filter_string="", run_view_type=ViewType.DELETED_ONLY
        )
        deleted_run_ids = []
        for deleted_run in deleted_runs:
            _, run_dir = self._find_run_root(deleted_run.info.run_uuid)
            meta = read_yaml(run_dir, FileStore.META_DATA_FILE_NAME)
            if "deleted_time" not in meta or current_time - int(meta["deleted_time"]) >= older_than:
                deleted_run_ids.append(deleted_run.info.run_uuid)

        return deleted_run_ids

    def restore_run(self, run_id):
        run_info = self._get_run_info(run_id)
        if run_info is None:
            raise MlflowException(
                "Run '%s' metadata is in invalid state." % run_id, databricks_pb2.INVALID_STATE
            )
        new_info = run_info._copy_with_overrides(lifecycle_stage=LifecycleStage.ACTIVE)
        self._overwrite_run_info(new_info, deleted_time=None)

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

    def update_run_info(self, run_id, run_status, end_time, run_name):
        _validate_run_id(run_id)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        new_info = run_info._copy_with_overrides(run_status, end_time, run_name=run_name)
        if run_name:
            self._set_run_tag(run_info, RunTag(MLFLOW_RUN_NAME, run_name))
        self._overwrite_run_info(new_info)
        return new_info

    def create_run(self, experiment_id, user_id, start_time, tags, run_name):
        """
        Creates a run with the specified attributes.
        """
        experiment_id = FileStore.DEFAULT_EXPERIMENT_ID if experiment_id is None else experiment_id
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise MlflowException(
                "Could not create run under experiment with ID %s - no such experiment "
                "exists." % experiment_id,
                databricks_pb2.RESOURCE_DOES_NOT_EXIST,
            )
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                "Could not create run under non-active experiment with ID %s." % experiment_id,
                databricks_pb2.INVALID_STATE,
            )
        tags = tags or []
        run_name_tag = _get_run_name_from_tags(tags)
        if run_name and run_name_tag and run_name != run_name_tag:
            raise MlflowException(
                "Both 'run_name' argument and 'mlflow.runName' tag are specified, but with "
                f"different values (run_name='{run_name}', mlflow.runName='{run_name_tag}').",
                INVALID_PARAMETER_VALUE,
            )
        run_name = run_name or run_name_tag or _generate_random_name()
        if not run_name_tag:
            tags.append(RunTag(key=MLFLOW_RUN_NAME, value=run_name))
        run_uuid = uuid.uuid4().hex
        artifact_uri = self._get_artifact_dir(experiment_id, run_uuid)
        run_info = RunInfo(
            run_uuid=run_uuid,
            run_id=run_uuid,
            run_name=run_name,
            experiment_id=experiment_id,
            artifact_uri=artifact_uri,
            user_id=user_id,
            status=RunStatus.to_string(RunStatus.RUNNING),
            start_time=start_time,
            end_time=None,
            lifecycle_stage=LifecycleStage.ACTIVE,
        )
        # Persist run metadata and create directories for logging metrics, parameters, artifacts
        run_dir = self._get_run_dir(run_info.experiment_id, run_info.run_id)
        mkdir(run_dir)
        run_info_dict = _make_persisted_run_info_dict(run_info)
        run_info_dict["deleted_time"] = None
        write_yaml(run_dir, FileStore.META_DATA_FILE_NAME, run_info_dict)
        mkdir(run_dir, FileStore.METRICS_FOLDER_NAME)
        mkdir(run_dir, FileStore.PARAMS_FOLDER_NAME)
        mkdir(run_dir, FileStore.ARTIFACTS_FOLDER_NAME)
        for tag in tags:
            self.set_tag(run_uuid, tag)
        return self.get_run(run_id=run_uuid)

    def get_run(self, run_id):
        """
        Note: Will get both active and deleted runs.
        """
        _validate_run_id(run_id)
        run_info = self._get_run_info(run_id)
        if run_info is None:
            raise MlflowException(
                "Run '%s' metadata is in invalid state." % run_id, databricks_pb2.INVALID_STATE
            )
        return self._get_run_from_info(run_info)

    def _get_run_from_info(self, run_info):
        metrics = self._get_all_metrics(run_info)
        params = self._get_all_params(run_info)
        tags = self._get_all_tags(run_info)
        if not run_info.run_name:
            run_name = _get_run_name_from_tags(tags)
            if run_name:
                run_info._set_run_name(run_name)
        return Run(run_info, RunData(metrics, params, tags))

    def _get_run_info(self, run_uuid):
        """
        Note: Will get both active and deleted runs.
        """
        exp_id, run_dir = self._find_run_root(run_uuid)
        if run_dir is None:
            raise MlflowException(
                "Run '%s' not found" % run_uuid, databricks_pb2.RESOURCE_DOES_NOT_EXIST
            )
        run_info = self._get_run_info_from_dir(run_dir)
        if run_info.experiment_id != exp_id:
            raise MlflowException(
                "Run '%s' metadata is in invalid state." % run_uuid, databricks_pb2.INVALID_STATE
            )
        return run_info

    def _get_run_info_from_dir(self, run_dir):
        meta = FileStore._read_yaml(run_dir, FileStore.META_DATA_FILE_NAME)
        run_info = _read_persisted_run_info_dict(meta)
        return run_info

    def _get_run_files(self, run_info, resource_type):
        run_dir = self._get_run_dir(run_info.experiment_id, run_info.run_id)
        # run_dir exists since run validity has been confirmed above.
        if resource_type == "metric":
            subfolder_name = FileStore.METRICS_FOLDER_NAME
        elif resource_type == "param":
            subfolder_name = FileStore.PARAMS_FOLDER_NAME
        elif resource_type == "tag":
            subfolder_name = FileStore.TAGS_FOLDER_NAME
        else:
            raise Exception("Looking for unknown resource under run.")
        return self._get_resource_files(run_dir, subfolder_name)

    def _get_experiment_files(self, experiment_id):
        _validate_experiment_id(experiment_id)
        experiment_dir = self._get_experiment_path(experiment_id, assert_exists=True)
        return self._get_resource_files(experiment_dir, FileStore.EXPERIMENT_TAGS_FOLDER_NAME)

    def _get_resource_files(self, root_dir, subfolder_name):
        source_dirs = find(root_dir, subfolder_name, full_path=True)
        if len(source_dirs) == 0:
            return root_dir, []
        file_names = []
        for root, _, files in os.walk(source_dirs[0]):
            for name in files:
                abspath = os.path.join(root, name)
                file_names.append(os.path.relpath(abspath, source_dirs[0]))
        if sys.platform == "win32":
            # Turn metric relative path into metric name.
            # Metrics can have '/' in the name. On windows, '/' is interpreted as a separator.
            # When the metric is read back the path will use '\' for separator.
            # We need to translate the path into posix path.
            from mlflow.utils.file_utils import relative_path_to_artifact_path

            file_names = [relative_path_to_artifact_path(x) for x in file_names]
        return source_dirs[0], file_names

    @staticmethod
    def _get_metric_from_file(parent_path, metric_name):
        _validate_metric_name(metric_name)
        metric_objs = [
            FileStore._get_metric_from_line(metric_name, line)
            for line in read_file_lines(parent_path, metric_name)
        ]
        if len(metric_objs) == 0:
            raise ValueError("Metric '%s' is malformed. No data found." % metric_name)
        # Python performs element-wise comparison of equal-length tuples, ordering them
        # based on their first differing element. Therefore, we use max() operator to find the
        # largest value at the largest timestamp. For more information, see
        # https://docs.python.org/3/reference/expressions.html#value-comparisons
        return max(metric_objs, key=lambda m: (m.step, m.timestamp, m.value))

    def get_all_metrics(self, run_uuid):
        _validate_run_id(run_uuid)
        run_info = self._get_run_info(run_uuid)
        return self._get_all_metrics(run_info)

    def _get_all_metrics(self, run_info):
        parent_path, metric_files = self._get_run_files(run_info, "metric")
        metrics = []
        for metric_file in metric_files:
            metrics.append(self._get_metric_from_file(parent_path, metric_file))
        return metrics

    @staticmethod
    def _get_metric_from_line(metric_name, metric_line):
        metric_parts = metric_line.strip().split(" ")
        if len(metric_parts) != 2 and len(metric_parts) != 3:
            raise MlflowException(
                "Metric '%s' is malformed; persisted metric data contained %s "
                "fields. Expected 2 or 3 fields." % (metric_name, len(metric_parts)),
                databricks_pb2.INTERNAL_ERROR,
            )
        ts = int(metric_parts[0])
        val = float(metric_parts[1])
        step = int(metric_parts[2]) if len(metric_parts) == 3 else 0
        return Metric(key=metric_name, value=val, timestamp=ts, step=step)

    def get_metric_history(self, run_id, metric_key, max_results=None, page_token=None):
        """
        Return all logged values for a given metric.

        :param run_id: Unique identifier for run
        :param metric_key: Metric name within the run
        :param max_results: An indicator for paginated results. This functionality is not
            implemented for FileStore and is unused in this store's implementation.
        :param page_token: An indicator for paginated results. This functionality is not
            implemented for FileStore and if the value is overridden with a value other than
            ``None``, an MlflowException will be thrown.

        :return: A List of :py:class:`mlflow.entities.Metric` entities if ``metric_key`` values
            have been logged to the ``run_id``, else an empty list.
        """
        # NB: The FileStore does not currently support pagination for this API.
        # Raise if `page_token` is specified, as the functionality to support paged queries
        # is not implemented.
        if page_token is not None:
            raise MlflowException(
                "The FileStore backend does not support pagination for the "
                f"`get_metric_history` API. Supplied argument `page_token` '{page_token}' must "
                "be `None`."
            )

        _validate_run_id(run_id)
        _validate_metric_name(metric_key)
        run_info = self._get_run_info(run_id)

        parent_path, metric_files = self._get_run_files(run_info, "metric")
        if metric_key not in metric_files:
            return PagedList([], None)
        return PagedList(
            [
                FileStore._get_metric_from_line(metric_key, line)
                for line in read_file_lines(parent_path, metric_key)
            ],
            None,
        )

    @staticmethod
    def _get_param_from_file(parent_path, param_name):
        _validate_param_name(param_name)
        value = read_file(parent_path, param_name)
        return Param(param_name, value)

    def get_all_params(self, run_uuid):
        _validate_run_id(run_uuid)
        run_info = self._get_run_info(run_uuid)
        return self._get_all_params(run_info)

    def _get_all_params(self, run_info):
        parent_path, param_files = self._get_run_files(run_info, "param")
        params = []
        for param_file in param_files:
            params.append(self._get_param_from_file(parent_path, param_file))
        return params

    @staticmethod
    def _get_experiment_tag_from_file(parent_path, tag_name):
        _validate_tag_name(tag_name)
        tag_data = read_file(parent_path, tag_name)
        return ExperimentTag(tag_name, tag_data)

    def get_all_experiment_tags(self, exp_id):
        parent_path, tag_files = self._get_experiment_files(exp_id)
        tags = []
        for tag_file in tag_files:
            tags.append(self._get_experiment_tag_from_file(parent_path, tag_file))
        return tags

    @staticmethod
    def _get_tag_from_file(parent_path, tag_name):
        _validate_tag_name(tag_name)
        tag_data = read_file(parent_path, tag_name)
        return RunTag(tag_name, tag_data)

    def get_all_tags(self, run_uuid):
        _validate_run_id(run_uuid)
        run_info = self._get_run_info(run_uuid)
        return self._get_all_tags(run_info)

    def _get_all_tags(self, run_info):
        parent_path, tag_files = self._get_run_files(run_info, "tag")
        tags = []
        for tag_file in tag_files:
            tags.append(self._get_tag_from_file(parent_path, tag_file))
        return tags

    def _list_run_infos(self, experiment_id, view_type):
        self._check_root_dir()
        if not self._has_experiment(experiment_id):
            return []
        experiment_dir = self._get_experiment_path(experiment_id, assert_exists=True)
        run_dirs = list_all(
            experiment_dir,
            filter_func=lambda x: all(
                os.path.basename(os.path.normpath(x)) != reservedFolderName
                for reservedFolderName in FileStore.RESERVED_EXPERIMENT_FOLDERS
            )
            and os.path.isdir(x),
            full_path=True,
        )
        run_infos = []
        for r_dir in run_dirs:
            try:
                # trap and warn known issues, will raise unexpected exceptions to caller
                run_info = self._get_run_info_from_dir(r_dir)
                if run_info.experiment_id != experiment_id:
                    logging.warning(
                        "Wrong experiment ID (%s) recorded for run '%s'. "
                        "It should be %s. Run will be ignored.",
                        str(run_info.experiment_id),
                        str(run_info.run_id),
                        str(experiment_id),
                        exc_info=True,
                    )
                    continue
                if LifecycleStage.matches_view_type(view_type, run_info.lifecycle_stage):
                    run_infos.append(run_info)
            except MissingConfigException as rnfe:
                # trap malformed run exception and log warning
                r_id = os.path.basename(r_dir)
                logging.warning(
                    "Malformed run '%s'. Detailed error %s", r_id, str(rnfe), exc_info=True
                )
        return run_infos

    def _search_runs(
        self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token
    ):
        if max_results > SEARCH_MAX_RESULTS_THRESHOLD:
            raise MlflowException(
                "Invalid value for request parameter max_results. It must be at "
                f"most {SEARCH_MAX_RESULTS_THRESHOLD}, but got value {max_results}",
                databricks_pb2.INVALID_PARAMETER_VALUE,
            )
        runs = []
        for experiment_id in experiment_ids:
            run_infos = self._list_run_infos(experiment_id, run_view_type)
            runs.extend(self._get_run_from_info(r) for r in run_infos)
        filtered = SearchUtils.filter(runs, filter_string)
        sorted_runs = SearchUtils.sort(filtered, order_by)
        runs, next_page_token = SearchUtils.paginate(sorted_runs, page_token, max_results)
        return runs, next_page_token

    def log_metric(self, run_id, metric):
        _validate_run_id(run_id)
        _validate_metric(metric.key, metric.value, metric.timestamp, metric.step)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        self._log_run_metric(run_info, metric)

    def _log_run_metric(self, run_info, metric):
        metric_path = self._get_metric_path(run_info.experiment_id, run_info.run_id, metric.key)
        make_containing_dirs(metric_path)
        append_to(metric_path, "{} {} {}\n".format(metric.timestamp, metric.value, metric.step))

    def _writeable_value(self, tag_value):
        if tag_value is None:
            return ""
        elif is_string_type(tag_value):
            return tag_value
        else:
            return "%s" % tag_value

    def log_param(self, run_id, param):
        _validate_run_id(run_id)
        _validate_param(param.key, param.value)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        self._log_run_param(run_info, param)

    def _log_run_param(self, run_info, param):
        param_path = self._get_param_path(run_info.experiment_id, run_info.run_id, param.key)
        writeable_param_value = self._writeable_value(param.value)
        if os.path.exists(param_path):
            self._validate_new_param_value(
                param_path=param_path,
                param_key=param.key,
                run_id=run_info.run_id,
                new_value=writeable_param_value,
            )
        make_containing_dirs(param_path)
        write_to(param_path, writeable_param_value)

    def _validate_new_param_value(self, param_path, param_key, run_id, new_value):
        """
        When logging a parameter with a key that already exists, this function is used to
        enforce immutability by verifying that the specified parameter value matches the existing
        value.
        :raises: py:class:`mlflow.exceptions.MlflowException` if the specified new parameter value
                 does not match the existing parameter value.
        """
        with open(param_path) as param_file:
            current_value = param_file.read()
        if current_value != new_value:
            raise MlflowException(
                f"Changing param values is not allowed. Param with key='{param_key}' was already"
                f" logged with value='{current_value}' for run ID='{run_id}'. Attempted logging"
                f" new value '{new_value}'.",
                databricks_pb2.INVALID_PARAMETER_VALUE,
            )

    def set_experiment_tag(self, experiment_id, tag):
        """
        Set a tag for the specified experiment

        :param experiment_id: String ID of the experiment
        :param tag: ExperimentRunTag instance to log
        """
        _validate_tag_name(tag.key)
        experiment = self.get_experiment(experiment_id)
        if experiment.lifecycle_stage != LifecycleStage.ACTIVE:
            raise MlflowException(
                "The experiment {} must be in the 'active' "
                "lifecycle_stage to set tags".format(experiment.experiment_id),
                error_code=databricks_pb2.INVALID_PARAMETER_VALUE,
            )
        tag_path = self._get_experiment_tag_path(experiment_id, tag.key)
        make_containing_dirs(tag_path)
        write_to(tag_path, self._writeable_value(tag.value))

    def set_tag(self, run_id, tag):
        _validate_run_id(run_id)
        _validate_tag_name(tag.key)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        self._set_run_tag(run_info, tag)
        if tag.key == MLFLOW_RUN_NAME:
            run_status = RunStatus.from_string(run_info.status)
            self.update_run_info(run_id, run_status, run_info.end_time, tag.value)

    def _set_run_tag(self, run_info, tag):
        tag_path = self._get_tag_path(run_info.experiment_id, run_info.run_id, tag.key)
        make_containing_dirs(tag_path)
        # Don't add trailing newline
        write_to(tag_path, self._writeable_value(tag.value))

    def delete_tag(self, run_id, key):
        """
        Delete a tag from a run. This is irreversible.
        :param run_id: String ID of the run
        :param key: Name of the tag
        """
        _validate_run_id(run_id)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        tag_path = self._get_tag_path(run_info.experiment_id, run_id, key)
        if not exists(tag_path):
            raise MlflowException(
                f"No tag with name: {key} in run with id {run_id}",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        os.remove(tag_path)

    def _overwrite_run_info(self, run_info, deleted_time=None):
        run_dir = self._get_run_dir(run_info.experiment_id, run_info.run_id)
        run_info_dict = _make_persisted_run_info_dict(run_info)
        if deleted_time is not None:
            run_info_dict["deleted_time"] = deleted_time
        write_yaml(run_dir, FileStore.META_DATA_FILE_NAME, run_info_dict, overwrite=True)

    def log_batch(self, run_id, metrics, params, tags):
        _validate_run_id(run_id)
        _validate_batch_log_data(metrics, params, tags)
        _validate_batch_log_limits(metrics, params, tags)
        _validate_param_keys_unique(params)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        try:
            for param in params:
                self._log_run_param(run_info, param)
            for metric in metrics:
                self._log_run_metric(run_info, metric)
            for tag in tags:
                # NB: If the tag run name value is set, update the run info to assure
                # synchronization.
                if tag.key == MLFLOW_RUN_NAME:
                    run_status = RunStatus.from_string(run_info.status)
                    self.update_run_info(run_id, run_status, run_info.end_time, tag.value)
                self._set_run_tag(run_info, tag)
        except Exception as e:
            raise MlflowException(e, INTERNAL_ERROR)

    def record_logged_model(self, run_id, mlflow_model):
        from mlflow.models import Model

        if not isinstance(mlflow_model, Model):
            raise TypeError(
                "Argument 'mlflow_model' should be mlflow.models.Model, got '{}'".format(
                    type(mlflow_model)
                )
            )
        _validate_run_id(run_id)
        run_info = self._get_run_info(run_id)
        check_run_is_active(run_info)
        model_dict = mlflow_model.to_dict()
        run_info = self._get_run_info(run_id)
        path = self._get_tag_path(run_info.experiment_id, run_info.run_id, MLFLOW_LOGGED_MODELS)
        if os.path.exists(path):
            with open(path) as f:
                model_list = json.loads(f.read())
        else:
            model_list = []
        tag = RunTag(MLFLOW_LOGGED_MODELS, json.dumps(model_list + [model_dict]))

        try:
            self._set_run_tag(run_info, tag)
        except Exception as e:
            raise MlflowException(e, INTERNAL_ERROR)

    @staticmethod
    def _read_yaml(root, file_name, retries=2):
        """
        Read data from yaml file and return as dictionary, retrying up to
        a specified number of times if the file contents are unexpectedly
        empty due to a concurrent write.

        :param root: Directory name.
        :param file_name: File name. Expects to have '.yaml' extension.
        :param retries: The number of times to retry for unexpected empty content.
        :return: Data in yaml file as dictionary
        """

        def _read_helper(root, file_name, attempts_remaining=2):
            result = read_yaml(root, file_name)
            if result is not None or attempts_remaining == 0:
                return result
            else:
                time.sleep(0.1 * (3 - attempts_remaining))
                return _read_helper(root, file_name, attempts_remaining - 1)

        return _read_helper(root, file_name, attempts_remaining=retries)
