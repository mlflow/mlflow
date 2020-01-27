from mlflow.entities import Experiment, Run, RunInfo, Metric, ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.service_pb2 import CreateExperiment, MlflowService, GetExperiment, \
    GetRun, SearchRuns, ListExperiments, GetMetricHistory, LogMetric, LogParam, SetTag, \
    UpdateRun, CreateRun, DeleteRun, RestoreRun, DeleteExperiment, RestoreExperiment, \
    UpdateExperiment, LogBatch, DeleteTag, SetExperimentTag, GetExperimentByName
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import call_endpoint, extract_api_info_for_service

_PATH_PREFIX = "/api/2.0"
_METHOD_TO_INFO = extract_api_info_for_service(MlflowService, _PATH_PREFIX)


class RestStore(AbstractStore):
    """
    Client for a remote tracking server accessed via REST API calls

    :param get_host_creds: Method to be invoked prior to every REST request to get the
      :py:class:`mlflow.rest_utils.MlflowHostCreds` for the request. Note that this
      is a function so that we can obtain fresh credentials in the case of expiry.
    """

    def __init__(self, get_host_creds):
        super(RestStore, self).__init__()
        self.get_host_creds = get_host_creds

    def _call_endpoint(self, api, json_body):
        endpoint, method = _METHOD_TO_INFO[api]
        response_proto = api.Response()
        return call_endpoint(self.get_host_creds(), endpoint, method, json_body, response_proto)

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        """
        :return: a list of all known Experiment objects
        """
        req_body = message_to_json(ListExperiments(view_type=view_type))
        response_proto = self._call_endpoint(ListExperiments, req_body)
        return [Experiment.from_proto(experiment_proto)
                for experiment_proto in response_proto.experiments]

    def create_experiment(self, name, artifact_location=None):
        """
        Create a new experiment.
        If an experiment with the given name already exists, throws exception.

        :param name: Desired name for an experiment

        :return: experiment_id (string) for the newly created experiment if successful, else None
        """
        req_body = message_to_json(CreateExperiment(
            name=name, artifact_location=artifact_location))
        response_proto = self._call_endpoint(CreateExperiment, req_body)
        return response_proto.experiment_id

    def get_experiment(self, experiment_id):
        """
        Fetch the experiment from the backend store.

        :param experiment_id: String id for the experiment

        :return: A single :py:class:`mlflow.entities.Experiment` object if it exists,
        otherwise raises an Exception.
        """
        req_body = message_to_json(GetExperiment(experiment_id=str(experiment_id)))
        response_proto = self._call_endpoint(GetExperiment, req_body)
        return Experiment.from_proto(response_proto.experiment)

    def delete_experiment(self, experiment_id):
        req_body = message_to_json(DeleteExperiment(experiment_id=str(experiment_id)))
        self._call_endpoint(DeleteExperiment, req_body)

    def restore_experiment(self, experiment_id):
        req_body = message_to_json(RestoreExperiment(experiment_id=str(experiment_id)))
        self._call_endpoint(RestoreExperiment, req_body)

    def rename_experiment(self, experiment_id, new_name):
        req_body = message_to_json(UpdateExperiment(
            experiment_id=str(experiment_id), new_name=new_name))
        self._call_endpoint(UpdateExperiment, req_body)

    def get_run(self, run_id):
        """
        Fetch the run from backend store

        :param run_id: Unique identifier for the run

        :return: A single Run object if it exists, otherwise raises an Exception
        """
        req_body = message_to_json(GetRun(run_uuid=run_id, run_id=run_id))
        response_proto = self._call_endpoint(GetRun, req_body)
        return Run.from_proto(response_proto.run)

    def update_run_info(self, run_id, run_status, end_time):
        """ Updates the metadata of the specified run. """
        req_body = message_to_json(UpdateRun(run_uuid=run_id, run_id=run_id, status=run_status,
                                             end_time=end_time))
        response_proto = self._call_endpoint(UpdateRun, req_body)
        return RunInfo.from_proto(response_proto.run_info)

    def create_run(self, experiment_id, user_id, start_time, tags):
        """
        Create a run under the specified experiment ID, setting the run's status to "RUNNING"
        and the start time to the current time.

        :param experiment_id: ID of the experiment for this run
        :param user_id: ID of the user launching this run
        :param source_type: Enum (integer) describing the source of the run

        :return: The created Run object
        """
        tag_protos = [tag.to_proto() for tag in tags]
        req_body = message_to_json(CreateRun(
            experiment_id=str(experiment_id), user_id=user_id,
            start_time=start_time, tags=tag_protos))
        response_proto = self._call_endpoint(CreateRun, req_body)
        run = Run.from_proto(response_proto.run)
        return run

    def log_metric(self, run_id, metric):
        """
        Log a metric for the specified run

        :param run_id: String id for the run
        :param metric: Metric instance to log
        """
        req_body = message_to_json(LogMetric(
            run_uuid=run_id, run_id=run_id,
            key=metric.key, value=metric.value, timestamp=metric.timestamp,
            step=metric.step))
        self._call_endpoint(LogMetric, req_body)

    def log_param(self, run_id, param):
        """
        Log a param for the specified run

        :param run_id: String id for the run
        :param param: Param instance to log
        """
        req_body = message_to_json(LogParam(
            run_uuid=run_id, run_id=run_id, key=param.key, value=param.value))
        self._call_endpoint(LogParam, req_body)

    def set_experiment_tag(self, experiment_id, tag):
        """
        Set a tag for the specified experiment

        :param experiment_id: String ID of the experiment
        :param tag: ExperimentRunTag instance to log
        """
        req_body = message_to_json(SetExperimentTag(
            experiment_id=experiment_id, key=tag.key, value=tag.value))
        self._call_endpoint(SetExperimentTag, req_body)

    def set_tag(self, run_id, tag):
        """
        Set a tag for the specified run

        :param run_id: String ID of the run
        :param tag: RunTag instance to log
        """
        req_body = message_to_json(SetTag(
            run_uuid=run_id, run_id=run_id, key=tag.key, value=tag.value))
        self._call_endpoint(SetTag, req_body)

    def delete_tag(self, run_id, key):
        """
        Delete a tag from a run. This is irreversible.
        :param run_id: String ID of the run
        :param key: Name of the tag
        """
        req_body = message_to_json(DeleteTag(run_id=run_id, key=key))
        self._call_endpoint(DeleteTag, req_body)

    def get_metric_history(self, run_id, metric_key):
        """
        Return all logged values for a given metric.

        :param run_id: Unique identifier for run
        :param metric_key: Metric name within the run

        :return: A list of :py:class:`mlflow.entities.Metric` entities if logged, else empty list
        """
        req_body = message_to_json(GetMetricHistory(
            run_uuid=run_id, run_id=run_id, metric_key=metric_key))
        response_proto = self._call_endpoint(GetMetricHistory, req_body)
        return [Metric.from_proto(metric) for metric in response_proto.metrics]

    def _search_runs(self, experiment_ids, filter_string, run_view_type, max_results, order_by,
                     page_token):
        experiment_ids = [str(experiment_id) for experiment_id in experiment_ids]
        sr = SearchRuns(experiment_ids=experiment_ids,
                        filter=filter_string,
                        run_view_type=ViewType.to_proto(run_view_type),
                        max_results=max_results,
                        order_by=order_by,
                        page_token=page_token)
        req_body = message_to_json(sr)
        response_proto = self._call_endpoint(SearchRuns, req_body)
        runs = [Run.from_proto(proto_run) for proto_run in response_proto.runs]
        # If next_page_token is not set, we will see it as "". We need to convert this to None.
        next_page_token = None
        if response_proto.next_page_token:
            next_page_token = response_proto.next_page_token
        return runs, next_page_token

    def delete_run(self, run_id):
        req_body = message_to_json(DeleteRun(run_id=run_id))
        self._call_endpoint(DeleteRun, req_body)

    def restore_run(self, run_id):
        req_body = message_to_json(RestoreRun(run_id=run_id))
        self._call_endpoint(RestoreRun, req_body)

    def get_experiment_by_name(self, experiment_name):
        try:
            req_body = message_to_json(GetExperimentByName(experiment_name=experiment_name))
            response_proto = self._call_endpoint(GetExperimentByName, req_body)
            return Experiment.from_proto(response_proto.experiment)
        except MlflowException as e:
            if e.error_code == databricks_pb2.ErrorCode.Name(
                    databricks_pb2.RESOURCE_DOES_NOT_EXIST):
                return None
            elif e.error_code == databricks_pb2.ErrorCode.Name(
                    databricks_pb2.REQUEST_LIMIT_EXCEEDED):
                raise e
            # Fall back to using ListExperiments-based implementation.
            for experiment in self.list_experiments(ViewType.ALL):
                if experiment.name == experiment_name:
                    return experiment
            return None

    def log_batch(self, run_id, metrics, params, tags):
        metric_protos = [metric.to_proto() for metric in metrics]
        param_protos = [param.to_proto() for param in params]
        tag_protos = [tag.to_proto() for tag in tags]
        req_body = message_to_json(
            LogBatch(metrics=metric_protos, params=param_protos, tags=tag_protos, run_id=run_id))
        self._call_endpoint(LogBatch, req_body)


class DatabricksRestStore(RestStore):
    """
    Databricks-specific RestStore implementation that provides different fallback
    behavior when hitting the GetExperimentByName REST API fails - in particular, we only
    fall back to ListExperiments when the server responds with ENDPOINT_NOT_FOUND, rather than
    on all internal server errors. This implementation should be deprecated once
    GetExperimentByName is available everywhere.
    """
    def get_experiment_by_name(self, experiment_name):
        try:
            req_body = message_to_json(GetExperimentByName(experiment_name=experiment_name))
            response_proto = self._call_endpoint(GetExperimentByName, req_body)
            return Experiment.from_proto(response_proto.experiment)
        except MlflowException as e:
            if e.error_code == databricks_pb2.ErrorCode.Name(
                    databricks_pb2.RESOURCE_DOES_NOT_EXIST):
                return None
            elif e.error_code == databricks_pb2.ErrorCode.Name(
                    databricks_pb2.ENDPOINT_NOT_FOUND):
                # Fall back to using ListExperiments-based implementation.
                for experiment in self.list_experiments(ViewType.ALL):
                    if experiment.name == experiment_name:
                        return experiment
                return None
            raise e
