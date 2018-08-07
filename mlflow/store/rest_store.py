import json
from google.protobuf.json_format import MessageToJson, ParseDict


from mlflow.store.abstract_store import AbstractStore

from mlflow.entities.experiment import Experiment
from mlflow.entities.run import Run
from mlflow.entities.run_info import RunInfo
from mlflow.entities.param import Param

from mlflow.entities.metric import Metric

from mlflow.utils.rest_utils import http_request

from mlflow.protos.service_pb2 import CreateExperiment, MlflowService, GetExperiment, \
    GetRun, SearchRuns, ListExperiments, GetMetricHistory, LogMetric, LogParam, UpdateRun,\
    CreateRun, GetMetric, GetParam

from mlflow.protos import databricks_pb2


def _get_path(endpoint_path):
    return "/api/2.0{}".format(endpoint_path)


def _api_method_to_info():
    """ Returns a dictionary mapping each API method to a tuple (path, HTTP method)"""
    service_methods = MlflowService.DESCRIPTOR.methods
    res = {}
    for service_method in service_methods:
        endpoints = service_method.GetOptions().Extensions[databricks_pb2.rpc].endpoints
        endpoint = endpoints[0]
        endpoint_path = _get_path(endpoint.path)
        res[MlflowService().GetRequestClass(service_method)] = (endpoint_path, endpoint.method)
    return res


_METHOD_TO_INFO = _api_method_to_info()


def _message_to_json(message):
    # preserving_proto_field_name keeps the JSON-serialized form snake_case
    return MessageToJson(message, preserving_proto_field_name=True)


class RestException(Exception):
    """Exception thrown on 400-level errors from the REST API"""
    def __init__(self, json):
        message = json['error_code']
        if 'message' in json:
            message = "%s: %s" % (message, json['message'])
        super(RestException, self).__init__(message)
        self.json = json


class RestStore(AbstractStore):
    """
    Client for a remote tracking server accessed via REST API calls
    :param http_request_kwargs arguments to add to rest_utils.http_request for all requests.
                               'hostname' is required.
    """

    def __init__(self, http_request_kwargs):
        super(RestStore, self).__init__()
        self.http_request_kwargs = http_request_kwargs
        if not http_request_kwargs['hostname']:
            raise Exception('hostname must be provided to RestStore')

    def _call_endpoint(self, api, json_body):
        endpoint, method = _METHOD_TO_INFO[api]
        response_proto = api.Response()
        # Convert json string to json dictionary, to pass to requests
        if json_body:
            json_body = json.loads(json_body)
        response = http_request(endpoint=endpoint, method=method,
                                json=json_body, **self.http_request_kwargs)
        js_dict = json.loads(response.text)

        if 'error_code' in js_dict:
            raise RestException(js_dict)

        ParseDict(js_dict=js_dict, message=response_proto)
        return response_proto

    def list_experiments(self):
        """
        :return: a list of all known Experiment objects
        """
        response_proto = self._call_endpoint(ListExperiments, None)
        return [Experiment.from_proto(experiment_proto)
                for experiment_proto in response_proto.experiments]

    def create_experiment(self, name, artifact_location=None):
        """
        Creates a new experiment.
        If an experiment with the given name already exists, throws exception.

        :param name: Desired name for an experiment
        :return: experiment_id (integer) for the newly created experiment if successful, else None
        """
        req_body = _message_to_json(CreateExperiment(
            name=name, artifact_location=artifact_location))
        response_proto = self._call_endpoint(CreateExperiment, req_body)
        return response_proto.experiment_id

    def get_experiment(self, experiment_id):
        """
        Fetches the experiment from the backend store.

        :param experiment_id: Integer id for the experiment
        :return: A single Experiment object if it exists, otherwise raises an Exception.
        """
        req_body = _message_to_json(GetExperiment(experiment_id=experiment_id))
        response_proto = self._call_endpoint(GetExperiment, req_body)
        return Experiment.from_proto(response_proto.experiment)

    def get_run(self, run_uuid):
        """
        Fetches the run from backend store

        :param run_uuid: Unique identifier for the run
        :return: A single Run object if it exists, otherwise raises an Exception
        """
        req_body = _message_to_json(GetRun(run_uuid=run_uuid))
        response_proto = self._call_endpoint(GetRun, req_body)
        return Run.from_proto(response_proto.run)

    def update_run_info(self, run_uuid, run_status, end_time):
        """ Updates the metadata of the specified run. """
        req_body = _message_to_json(UpdateRun(run_uuid=run_uuid, status=run_status,
                                              end_time=end_time))
        response_proto = self._call_endpoint(UpdateRun, req_body)
        return RunInfo.from_proto(response_proto.run_info)

    def create_run(self, experiment_id, user_id, run_name, source_type, source_name,
                   entry_point_name, start_time, source_version, tags):
        """
        Creates a run under the specified experiment ID, setting the run's status to "RUNNING"
        and the start time to the current time.

        :param experiment_id: ID of the experiment for this run
        :param user_id: ID of the user launching this run
        :param source_type: Enum (integer) describing the source of the run
        :return: The created Run object
        """
        tag_protos = [tag.to_proto() for tag in tags]
        req_body = _message_to_json(CreateRun(
            experiment_id=experiment_id, user_id=user_id, run_name=run_name,
            source_type=source_type, source_name=source_name, entry_point_name=entry_point_name,
            start_time=start_time, source_version=source_version, tags=tag_protos))
        response_proto = self._call_endpoint(CreateRun, req_body)
        return Run.from_proto(response_proto.run)

    def log_metric(self, run_uuid, metric):
        """
        Logs a metric for the specified run
        :param run_uuid: String id for the run
        :param metric: Metric instance to log
        """
        req_body = _message_to_json(LogMetric(
            run_uuid=run_uuid, key=metric.key, value=metric.value, timestamp=metric.timestamp))
        self._call_endpoint(LogMetric, req_body)

    def log_param(self, run_uuid, param):
        """
        Logs a param for the specified run
        :param run_uuid: String id for the run
        :param param: Param instance to log
        """
        req_body = _message_to_json(LogParam(run_uuid=run_uuid, key=param.key, value=param.value))
        self._call_endpoint(LogParam, req_body)

    def get_metric(self, run_uuid, metric_key):
        """
        Returns the last logged value for a given metric.

        :param run_uuid: Unique identifier for run
        :param metric_key: Metric name within the run

        :return: A single float value for the give metric if logged, else None
        """
        req_body = _message_to_json(GetMetric(run_uuid=run_uuid, metric_key=metric_key))
        response_proto = self._call_endpoint(GetMetric, req_body)
        return Metric.from_proto(response_proto.metric)

    def get_param(self, run_uuid, param_name):
        """
        Returns the value of the specified parameter.

        :param run_uuid: Unique identifier for run
        :param param_name: Parameter name within the run

        :return: Value of the given parameter if logged, else None
        """
        req_body = _message_to_json(GetParam(run_uuid=run_uuid, param_name=param_name))
        response_proto = self._call_endpoint(GetParam, req_body)
        return Param.from_proto(response_proto.parameter)

    def get_metric_history(self, run_uuid, metric_key):
        """
        Returns all logged value for a given metric.

        :param run_uuid: Unique identifier for run
        :param metric_key: Metric name within the run

        :return: A list of float values logged for the give metric if logged, else empty list
        """
        req_body = _message_to_json(GetMetricHistory(run_uuid=run_uuid, metric_key=metric_key))
        response_proto = self._call_endpoint(GetMetricHistory, req_body)
        return [Metric.from_proto(metric).value for metric in response_proto.metrics]

    def search_runs(self, experiment_ids, search_expressions):
        """
        Returns runs that match the given list of search expressions within the experiments.
        Given multiple search expressions, all these expressions are ANDed together for search.

        :param experiment_ids: List of experiment ids to scope the search
        :param search_expression: list of search expressions

        :return: A list of Run objects that satisfy the search expressions
        """
        search_expressions_protos = [expr.to_proto() for expr in search_expressions]
        req_body = _message_to_json(SearchRuns(experiment_ids=experiment_ids,
                                               search_expressions=search_expressions_protos))
        response_proto = self._call_endpoint(SearchRuns, req_body)
        return [Run.from_proto(proto_run) for proto_run in response_proto.runs]

    def list_run_infos(self, experiment_id):
        """
        Returns run information for runs which belong to the experiment_id

        :param experiment_id: The experiment id which to search.

        :return: A list of RunInfo objects that satisfy the search expressions
        """
        runs = self.search_runs(experiment_ids=[experiment_id], search_expressions=[])
        return [run.info for run in runs]


class DatabricksStore(RestStore):
    """
    A specific type of RestStore which includes authentication information to Databricks.
    :param http_request_kwargs arguments to add to rest_utils.http_request for all requests.
                               'hostname', 'headers', and 'secure_verify' are required.
    """
    def __init__(self, http_request_kwargs):
        if http_request_kwargs['hostname'] is None:
            raise Exception('hostname must be provided to DatabricksStore')
        if http_request_kwargs['headers'] is None:
            raise Exception('headers must be provided to DatabricksStore')
        if http_request_kwargs['verify'] is None:
            raise Exception('verify must be provided to DatabricksStore')
        super(DatabricksStore, self).__init__(http_request_kwargs)
