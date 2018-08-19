"""
Internal package providing a Python CRUD interface to MLflow Experiments and Runs.
This is a lower level API than the :py:mod:`mlflow.tracking.fluent` module, and is
exposed to users in the :py:mod:`mlflow.tracking` module.
"""

import os
import time
from six import iteritems

from mlflow.utils.validation import _validate_metric_name, _validate_param_name, _validate_run_id
from mlflow.entities import Param, Metric, Run, RunStatus, RunTag
from mlflow.tracking.utils import _get_store
from mlflow.store.artifact_repo import ArtifactRepository

_DEFAULT_USER_ID = "unknown"


class MLflowService(object):
    """Client to an MLflow Tracking Server that can create and manage experiments and
    runs. This may either manage files locally or remotely depending on the
    AbstractStore provided.
    """

    def __init__(self, store):
        self.store = store

    def get_run(self, run_id):
        """:return: :py:class:`mlflow.entities.Run` associated with this run id"""
        _validate_run_id(run_id)
        return self.store.get_run(run_id)

    def create_run(self, experiment_id, user_id=None, run_name=None, source_type=None,
                   source_name=None, entry_point_name=None, start_time=None,
                   source_version=None, tags=None):
        """Creates a new :py:class:`mlflow.entities.Run` object, which can be associated with
        metrics, parameters, artifacts, etc.
        Unlike :py:func:`mlflow.projects.run`, does not actually run code, just creates objects.
        Unlike :py:func:`mlflow.start_run`, this does not change the "active run" used by
        :py:func:`mlflow.log_param` and friends.

        :param user_id: If not provided, we will use the current user as a default.
        :param start_time: If not provided, we will use the current timestamp.
        :param tags: A dictionary of key-value pairs which will be converted into
          RunTag objects.
        :return: :py:class:`mlflow.entities.Run` which was created
        """
        tags = tags if tags else {}
        return self.store.create_run(
            experiment_id=experiment_id,
            user_id=user_id if user_id is not None else _get_user_id(),
            run_name=run_name,
            source_type=source_type,
            source_name=source_name,
            entry_point_name=entry_point_name,
            start_time=start_time or int(time.time() * 1000),
            source_version=source_version,
            tags=[RunTag(key, value) for (key, value) in iteritems(tags)],
        )

    def list_runs(self, experiment_id):
        """:return: list of :py:class:`mlflow.entities.Run` (with only RunInfo filled)"""
        run_infos = self.store.list_run_infos(experiment_id)
        return [Run(run_info.run_uuid, run_info) for run_info in run_infos]

    def list_experiments(self):
        """:return: list of :py:class:`mlflow.entities.Experiment`"""
        return self.store.list_experiments()

    def get_experiment(self, experiment_id):
        """:return: :py:class:`mlflow.entities.Experiment`"""
        return self.store.get_experiment(experiment_id)

    def create_experiment(self, name, artifact_location=None):
        """Creates an experiment.

        :param name: must be unique
        :param artifact_location: If not provided, the server will pick an appropriate default.
        :return: integer id of the created experiment
        """
        return self.store.create_experiment(
            name=name,
            artifact_location=artifact_location,
        )

    def log_metric(self, run_id, key, value, timestamp=None):
        """Logs a metric against the given run id. If timestamp is not provided, we will
        use the current timestamp.
        """
        _validate_metric_name(key)
        timestamp = timestamp if timestamp is not None else int(time.time())
        metric = Metric(key, value, timestamp)
        self.store.log_metric(run_id, metric)

    def log_param(self, run_id, key, value):
        """Logs a parameter against the given run id. Value will be converted to a string."""
        _validate_param_name(key)
        param = Param(key, str(value))
        self.store.log_param(run_id, param)

    def log_artifact(self, artifact_uri, local_path, artifact_path=None):
        """Writes a local file to the remote artifact_uri.

        :param local_path: of the file to write
        :param artifact_path: If provided, will be directory in artifact_uri to write to"""
        artifact_repo = ArtifactRepository.from_artifact_uri(artifact_uri, self.store)
        artifact_repo.log_artifact(local_path, artifact_path)

    def log_artifacts(self, artifact_uri, local_dir, artifact_path=None):
        """Writes a directory of files to the remote artifact_uri.

        :param local_dir: of the file to write
        :param artifact_path: If provided, will be directory in artifact_uri to write to"""
        artifact_repo = ArtifactRepository.from_artifact_uri(artifact_uri, self.store)
        artifact_repo.log_artifacts(local_dir, artifact_path)

    def set_terminated(self, run_id, status=None, end_time=None):
        """Sets a Run's status to terminated

        :param status: A string value of :py:class:`mlflow.entities.RunStatus`.
          Defaults to FINISHED.
        :param end_time: If not provided, defaults to the current time."""
        end_time = end_time if end_time else int(time.time() * 1000)
        status = status if status else "FINISHED"
        self.store.update_run_info(run_id, run_status=RunStatus.from_string(status),
                                   end_time=end_time)


def get_service(tracking_uri=None):
    """
    :param tracking_uri: Address of local or remote tracking server. If not provided,
      this will default to the store set by mlflow.tracking.set_tracking_uri. See
      https://mlflow.org/docs/latest/tracking.html#where-runs-get-recorded for more info.
    :return: mlflow.tracking.MLflowService"""
    store = _get_store(tracking_uri)
    return MLflowService(store)


def _get_user_id():
    """Get the ID of the user for the current run."""
    try:
        import pwd
        return pwd.getpwuid(os.getuid())[0]
    except ImportError:
        return _DEFAULT_USER_ID
