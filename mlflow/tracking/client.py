"""
Internal package providing a Python CRUD interface to MLflow experiments and runs.
This is a lower level API than the :py:mod:`mlflow.tracking.fluent` module, and is
exposed in the :py:mod:`mlflow.tracking` module.
"""

import os
import time
from six import iteritems

from mlflow.tracking import utils
from mlflow.utils.validation import _validate_param_name, _validate_tag_name, _validate_run_id, \
    _validate_experiment_name, _validate_metric
from mlflow.entities import Param, Metric, RunStatus, RunTag, ViewType, SourceType
from mlflow.store.artifact_repository_registry import get_artifact_repository
from mlflow.utils.mlflow_tags import MLFLOW_SOURCE_NAME, MLFLOW_SOURCE_TYPE, MLFLOW_PARENT_RUN_ID, \
    MLFLOW_GIT_COMMIT, MLFLOW_PROJECT_ENTRY_POINT

_DEFAULT_USER_ID = "unknown"


class MlflowClient(object):
    """Client of an MLflow Tracking Server that creates and manages experiments and runs.
    """

    def __init__(self, tracking_uri=None):
        """
        :param tracking_uri: Address of local or remote tracking server. If not provided, defaults
                             to the service set by ``mlflow.tracking.set_tracking_uri``. See
                             `Where Runs Get Recorded <../tracking.html#where-runs-get-recorded>`_
                             for more info.
        """
        self.tracking_uri = tracking_uri or utils.get_tracking_uri()
        self.store = utils._get_store(self.tracking_uri)

    def get_run(self, run_id):
        """:return: :py:class:`mlflow.entities.Run` associated with the run ID."""
        _validate_run_id(run_id)
        return self.store.get_run(run_id)

    def create_run(self, experiment_id, user_id=None, run_name=None, start_time=None, tags=None):
        """
        Create a :py:class:`mlflow.entities.Run` object that can be associated with
        metrics, parameters, artifacts, etc.
        Unlike :py:func:`mlflow.projects.run`, creates objects but does not run code.
        Unlike :py:func:`mlflow.start_run`, does not change the "active run" used by
        :py:func:`mlflow.log_param`.

        :param user_id: If not provided, use the current user as a default.
        :param start_time: If not provided, use the current timestamp.
        :param tags: A dictionary of key-value pairs that are converted into
                     :py:class:`mlflow.entities.RunTag` objects.
        :return: :py:class:`mlflow.entities.Run` that was created.
        """

        tags = tags if tags else {}

        # Extract run attributes from tags
        # This logic is temporary; by the 1.0 release, this information will only be stored in tags
        # and will not be available as attributes of the run
        parent_run_id = tags.get(MLFLOW_PARENT_RUN_ID)
        source_name = tags.get(MLFLOW_SOURCE_NAME, "Python Application")
        source_version = tags.get(MLFLOW_GIT_COMMIT)
        entry_point_name = tags.get(MLFLOW_PROJECT_ENTRY_POINT)

        source_type_string = tags.get(MLFLOW_SOURCE_TYPE)
        if source_type_string is None:
            source_type = SourceType.LOCAL
        else:
            source_type = SourceType.from_string(source_type_string)

        return self.store.create_run(
            experiment_id=experiment_id,
            user_id=user_id if user_id is not None else _get_user_id(),
            run_name=run_name,
            start_time=start_time or int(time.time() * 1000),
            tags=[RunTag(key, value) for (key, value) in iteritems(tags)],
            # The below arguments remain set for backwards compatability:
            parent_run_id=parent_run_id,
            source_type=source_type,
            source_name=source_name,
            entry_point_name=entry_point_name,
            source_version=source_version
        )

    def list_run_infos(self, experiment_id, run_view_type=ViewType.ACTIVE_ONLY):
        """:return: List of :py:class:`mlflow.entities.RunInfo`"""
        return self.store.list_run_infos(experiment_id, run_view_type)

    def list_experiments(self):
        """:return: List of :py:class:`mlflow.entities.Experiment`"""
        return self.store.list_experiments()

    def get_experiment(self, experiment_id):
        """
        :param experiment_id: The experiment ID returned from ``create_experiment``.
        :return: :py:class:`mlflow.entities.Experiment`
        """
        return self.store.get_experiment(experiment_id)

    def get_experiment_by_name(self, name):
        """
        :param name: The experiment name.
        :return: :py:class:`mlflow.entities.Experiment`
        """
        return self.store.get_experiment_by_name(name)

    def create_experiment(self, name, artifact_location=None):
        """Create an experiment.

        :param name: The experiment name. Must be unique.
        :param artifact_location: The location to store run artifacts.
                                  If not provided, the server picks an appropriate default.
        :return: Integer ID of the created experiment.
        """
        _validate_experiment_name(name)
        return self.store.create_experiment(
            name=name,
            artifact_location=artifact_location,
        )

    def delete_experiment(self, experiment_id):
        """
        Delete an experiment from the backend store.

        :param experiment_id: The experiment ID returned from ``create_experiment``.
        """
        self.store.delete_experiment(experiment_id)

    def restore_experiment(self, experiment_id):
        """
        Restore a deleted experiment unless permanently deleted.

        :param experiment_id: The experiment ID returned from ``create_experiment``.
        """
        self.store.restore_experiment(experiment_id)

    def rename_experiment(self, experiment_id, new_name):
        """
        Update an experiment's name. The new name must be unique.

        :param experiment_id: The experiment ID returned from ``create_experiment``.
        """
        self.store.rename_experiment(experiment_id, new_name)

    def log_metric(self, run_id, key, value, timestamp=None):
        """
        Log a metric against the run ID. If timestamp is not provided, uses
        the current timestamp.
        """
        timestamp = timestamp if timestamp is not None else int(time.time())
        _validate_metric(key, value, timestamp)
        metric = Metric(key, value, timestamp)
        self.store.log_metric(run_id, metric)

    def log_param(self, run_id, key, value):
        """
        Log a parameter against the run ID. Value is converted to a string.
        """
        _validate_param_name(key)
        param = Param(key, str(value))
        self.store.log_param(run_id, param)

    def set_tag(self, run_id, key, value):
        """
        Set a tag on the run ID. Value is converted to a string.
        """
        _validate_tag_name(key)
        tag = RunTag(key, str(value))
        self.store.set_tag(run_id, tag)

    def log_batch(self, run_id, metrics, params, tags):
        """
        Log multiple metrics, params, and/or tags.

        :param metrics: List of Metric(key, value, timestamp) instances.
        :param params: List of Param(key, value) instances.
        :param tags: List of RunTag(key, value) instances.

        Raises an MlflowException if any errors occur.
        :returns: None
        """
        for metric in metrics:
            _validate_metric(metric.key, metric.value, metric.timestamp)
        for param in params:
            _validate_param_name(param.key)
        for tag in tags:
            _validate_tag_name(tag.key)
        self.store.log_batch(run_id=run_id, metrics=metrics, params=params, tags=tags)

    def log_artifact(self, run_id, local_path, artifact_path=None):
        """
        Write a local file to the remote ``artifact_uri``.

        :param local_path: Path to the file to write.
        :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.
        """
        run = self.get_run(run_id)
        artifact_repo = get_artifact_repository(run.info.artifact_uri, self.store)
        artifact_repo.log_artifact(local_path, artifact_path)

    def log_artifacts(self, run_id, local_dir, artifact_path=None):
        """
        Write a directory of files to the remote ``artifact_uri``.

        :param local_dir: Path to the directory of files to write.
        :param artifact_path: If provided, the directory in ``artifact_uri`` to write to.
        """
        run = self.get_run(run_id)
        artifact_repo = get_artifact_repository(run.info.artifact_uri, self.store)
        artifact_repo.log_artifacts(local_dir, artifact_path)

    def list_artifacts(self, run_id, path=None):
        """
        List the artifacts for a run.

        :param run_id: The run to list artifacts from.
        :param path: The run's relative artifact path to list from. By default it is set to None
                     or the root artifact path.
        :return: List of :py:class:`mlflow.entities.FileInfo`
        """
        run = self.get_run(run_id)
        artifact_root = run.info.artifact_uri
        artifact_repo = get_artifact_repository(artifact_root, self.store)
        return artifact_repo.list_artifacts(path)

    def download_artifacts(self, run_id, path):
        """
        Download an artifact file or directory from a run to a local directory if applicable,
        and return a local path for it.

        :param run_id: The run to download artifacts from.
        :param path: Relative source path to the desired artifact.
        :return: Local path of desired artifact.
        """
        run = self.get_run(run_id)
        artifact_root = run.info.artifact_uri
        artifact_repo = get_artifact_repository(artifact_root, self.store)
        return artifact_repo.download_artifacts(path)

    def set_terminated(self, run_id, status=None, end_time=None):
        """Set a run's status to terminated.

        :param status: A string value of :py:class:`mlflow.entities.RunStatus`.
                       Defaults to "FINISHED".
        :param end_time: If not provided, defaults to the current time."""
        end_time = end_time if end_time else int(time.time() * 1000)
        status = status if status else RunStatus.to_string(RunStatus.FINISHED)
        self.store.update_run_info(run_id, run_status=RunStatus.from_string(status),
                                   end_time=end_time)

    def delete_run(self, run_id):
        """
        Deletes a run with the given ID.
        """
        self.store.delete_run(run_id)

    def restore_run(self, run_id):
        """
        Restores a deleted run with the given ID.
        """
        self.store.restore_run(run_id)


def _get_user_id():
    """Get the ID of the user for the current run."""
    try:
        import pwd
        return pwd.getpwuid(os.getuid())[0]
    except ImportError:
        return _DEFAULT_USER_ID
