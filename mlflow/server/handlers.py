# Define all the service endpoint handlers here.
import json
import os
import re
import six

from flasgger import swag_from
from flask import Response, request, send_file
from google.protobuf.json_format import MessageToJson, ParseDict
from querystring_parser import parser

from mlflow.entities import Metric, Param, RunTag
from mlflow.protos import databricks_pb2
from mlflow.protos.service_pb2 import CreateExperiment, MlflowService, GetExperiment, \
    GetRun, SearchRuns, ListArtifacts, GetMetricHistory, CreateRun, \
    UpdateRun, LogMetric, LogParam, ListExperiments, GetMetric, GetParam
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.store.file_store import FileStore
from mlflow.utils.swagger import swagger_object_dict

_store = None


def _get_store():
    from mlflow.server import FILE_STORE_ENV_VAR, ARTIFACT_ROOT_ENV_VAR
    global _store
    if _store is None:
        store_dir = os.environ.get(FILE_STORE_ENV_VAR, os.path.abspath("mlruns"))
        artifact_root = os.environ.get(ARTIFACT_ROOT_ENV_VAR, store_dir)
        _store = FileStore(store_dir, artifact_root)
    return _store


def _get_request_message(request_message, flask_request=request):
    if flask_request.method == 'GET' and len(flask_request.query_string) > 0:
        # This is a hack to make arrays of length 1 work with the parser.
        # for example experiment_ids%5B%5D=0 should be parsed to {experiment_ids: [0]}
        # but it gets parsed to {experiment_ids: 0}
        # but it doesn't. However, experiment_ids%5B0%5D=0 will get parsed to the right
        # result.
        query_string = re.sub('%5B%5D', '%5B0%5D', flask_request.query_string.decode("utf-8"))
        request_dict = parser.parse(query_string, normalized=True)
        ParseDict(request_dict, request_message)
        return request_message

    request_json = flask_request.get_json(force=True, silent=True)

    # Older clients may post their JSON double-encoded as strings, so the get_json
    # above actually converts it to a string. Therefore, we check this condition
    # (which we can tell for sure because any proper request should be a dictionary),
    # and decode it a second time.
    if isinstance(request_json, six.string_types):
        request_json = json.loads(request_json)

    # If request doesn't have json body then assume it's empty.
    if request_json is None:
        request_json = {}
    ParseDict(request_json, request_message)
    return request_message


def get_handler(request_class):
    """
    :param request_class: The type of protobuf message
    :return:
    """
    return HANDLERS.get(request_class, _not_implemented)


_TEXT_EXTENSIONS = ['txt', 'yaml', 'json', 'js', 'py', 'csv', 'md', 'rst', 'MLmodel', 'MLproject']


def get_artifact_handler():
    query_string = request.query_string.decode('utf-8')
    request_dict = parser.parse(query_string, normalized=True)
    run = _get_store().get_run(request_dict['run_uuid'])
    filename = os.path.abspath(_get_artifact_repo(run).download_artifacts(request_dict['path']))
    extension = os.path.splitext(filename)[-1].replace(".", "")
    if extension in _TEXT_EXTENSIONS:
        return send_file(filename, mimetype='text/plain')
    else:
        return send_file(filename)


def _not_implemented():
    response = Response()
    response.status_code = 404
    return response


def _message_to_json(message):
    # preserving_proto_field_name keeps the JSON-serialized form snake_case
    return MessageToJson(message, preserving_proto_field_name=True)

@swag_from(swagger_object_dict)
def _create_experiment():
    """
    Create an experiment with a name.
    Returns the ID of the newly created experiment.
    Validates that another experiment with the same name does not already exist and
    fails if another experiment with the same name already exists.
    ---
    parameters:
      - name: name
        description: Experiment name.
        in: path
        type: string
        required: true
      - name: artifact_location
        description: Location where all artifacts for this experiment are stored. If not provided,
            the remote server will select an appropriate default.
        in: path
        type: string
        required: false
    responses:
      200:
        description: Unique identifier for created experiment.
        name: experiment_id
        type: integer
        format: int64
    """
    request_message = _get_request_message(CreateExperiment())
    experiment_id = _get_store().create_experiment(request_message.name,
                                                   request_message.artifact_location)
    response_message = CreateExperiment.Response()
    response_message.experiment_id = experiment_id
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response



def _get_experiment():
    """
    Get experiment details.
    Get metadata for an experiment and a list of runs for the experiment.

    ---
    parameters:
      - name: experiment_id
        description: Identifier to get an experiment. This field is required.
        in: body
        required: true
        schema:
          $ref: '#/definitions/ExperimentIdQuerySchema'
    responses:
      200:
        description: Experiment details.
        name: experiment_details
        type: object
        schema:
          $ref: '#/definitions/Experiment_details'
    """
    request_message = _get_request_message(GetExperiment())
    response_message = GetExperiment.Response()
    response_message.experiment.MergeFrom(_get_store().get_experiment(request_message.experiment_id)
                                          .to_proto())
    run_info_entities = _get_store().list_run_infos(request_message.experiment_id)
    response_message.runs.extend([r.to_proto() for r in run_info_entities])
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _create_run():
    """
    Create a new run within an experiment.
    A run is usually a single execution of a machine learning or data ETL pipeline.
    MLflow uses runs to track Param, Metric, and RunTag, associated with a single execution.

    ---
    parameters:
      - name: runinfo
        description: RunInfo specifies the details of a run.
        in: body
        required: true
        schema:
          $ref: '#/definitions/RunInfoDataSchema'
    responses:
      200:
        description: Metadata of the newly created run.
        name: run_info
        type: object
        schema:
          $ref: '#/definitions/RunInfo'
    """
    request_message = _get_request_message(CreateRun())

    tags = [RunTag(tag.key, tag.value) for tag in request_message.tags]
    run = _get_store().create_run(
        experiment_id=request_message.experiment_id,
        user_id=request_message.user_id,
        run_name=request_message.run_name,
        source_type=request_message.source_type,
        source_name=request_message.source_name,
        entry_point_name=request_message.entry_point_name,
        start_time=request_message.start_time,
        source_version=request_message.source_version,
        tags=tags)

    response_message = CreateRun.Response()
    response_message.run.MergeFrom(run.to_proto())
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


# TODO: what is the response for this request?
def _update_run():
    """
    Update run.

    ---
    parameters:
      - $ref: '#/definitions/ParamQuerySchema'
    responses:
      200:
        description: Artifacts array
        name: artifacts
        schema:
          $ref: '#/definitions/RunInfo'
    """
    request_message = _get_request_message(UpdateRun())
    updated_info = _get_store().update_run_info(request_message.run_uuid, request_message.status,
                                                request_message.end_time)
    response_message = UpdateRun.Response(run_info=updated_info.to_proto())
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


# TODO: what is the response for this request?
def _log_metric():
    """
    Log a metric for a run. Metrics key-value pair that record a single float measure.
    During a single execution of a run, a particular metric can be logged several times.
    Backend keeps track of historical values along with timestamps.

    ---
    parameters:
      - name: metric
        description: Metric metadata to be logged.
        in: body
        required: true
        schema:
          $ref: '#/definitions/MetricDataSchema'
    responses:
      200:
        description: Metric details.
        name: experiment_details
        type: object
        schema:
          $ref: '#/definitions/Experiment_details'
    """
    request_message = _get_request_message(LogMetric())
    metric = Metric(request_message.key, request_message.value, request_message.timestamp)
    _get_store().log_metric(request_message.run_uuid, metric)
    response_message = LogMetric.Response()
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


# TODO: what is the response for this request?
def _log_param():
    """
    Log a parameter used for this run.
    Examples are params and hyperparameters used for ML training, or constant dates and values
    used in an ETL pipeline. A params is a STRING key-value pair. For a run, a single parameter
    is allowed to be logged only once.
    ---
    parameters:
      - name: parameter
        description: Parameter metadata to be logged.
        in: body
        required: true
        schema:
          $ref: '#/definitions/ParamDataSchema'
    responses:
      200:
        description: Metric details.
        name: experiment_details
        type: object
        schema:
          $ref: '#/definitions/Experiment_details'
    """
    request_message = _get_request_message(LogParam())
    param = Param(request_message.key, request_message.value)
    _get_store().log_param(request_message.run_uuid, param)
    response_message = LogParam.Response()
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _get_run():
    """
    Get metadata, params, tags, and metrics for run.
    Only last logged value for each metric is returned.

    ---
    parameters:
      - name: run_uuid
        description: Run UUID. This field is required.
        in: body
        required: true
        schema:
          $ref: '#/definitions/RunUUIDGet'
    responses:
      200:
        description: Run details.
        name: run
        type: object
        schema:
          $ref: '#/definitions/RunInfo'
    """
    request_message = _get_request_message(GetRun())
    response_message = GetRun.Response()
    response_message.run.MergeFrom(_get_store().get_run(request_message.run_uuid).to_proto())
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _search_runs():
    """
    Search for runs that satisfy expressions.
    Search expressions can use Metric and Param keys.
    ---
    parameters:
      - name: exeriment_ids
        description: Identifier to get an experiment.
        in: body
        required: true
        type: array
        items:
          type: integer
          format: int64
      - name: anded_expressions
        description: Expressions describing runs.
        in: body
        required: true
        type: array
        items:
          anyOf:
            - $ref: '#/definitions/ParameterSearchExpression'
            - $ref: '#/definitions/MetricSearchExpression'
    responses:
      200:
        description: Runs that match the search criteria.
        name: runs
        type: array
        schema:
          $ref: '#/definitions/Run'
    """
    request_message = _get_request_message(SearchRuns())
    response_message = SearchRuns.Response()
    run_entities = _get_store().search_runs(request_message.experiment_ids,
                                            request_message.anded_expressions)
    response_message.runs.extend([r.to_proto() for r in run_entities])
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _list_artifacts():
    """
    List artifacts.

    ---
    parameters:
      - $ref: '#/definitions/RunUUIDQuerySchema'
      - name: path
        description: The relative_path to the output base directory.
        in: body
        required: true
        type: string
    responses:
      200:
        description: Artifacts array
        name: artifacts
        schema:
          $ref: '#/definitions/Artifacts'
    """
    request_message = _get_request_message(ListArtifacts())
    response_message = ListArtifacts.Response()
    if request_message.HasField('path'):
        path = request_message.path
    else:
        path = None
    run = _get_store().get_run(request_message.run_uuid)
    artifact_entities = _get_artifact_repo(run).list_artifacts(path)
    response_message.files.extend([a.to_proto() for a in artifact_entities])
    response_message.root_uri = _get_artifact_repo(run).artifact_uri
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _get_metric_history():
    """
    Retrieve all logged values for a metric.
    ---
    parameters:
      - name: metric
        description: Metric identifier.
        in: body
        required: true
        schema:
          $ref: '#/definitions/MetricQuerySchema'
    responses:
      200:
        description: Metric historical values.
        name: metrics
        type: array
        schema:
          $ref: '#/definitions/Metric'
    """
    request_message = _get_request_message(GetMetricHistory())
    response_message = GetMetricHistory.Response()
    metric_entites = _get_store().get_metric_history(request_message.run_uuid,
                                                     request_message.metric_key)
    response_message.metrics.extend([m.to_proto() for m in metric_entites])
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


# TODO: verify why this request's body has metric_key instead of key
def _get_metric():
    """
    Retrieve the logged value for a metric during a run.
    For a run, if this metric is logged more than once, this API retrieves only
    the latest value logged.
    ---
    parameters:
      - name: metric
        description: Metric identifier.
        in: body
        required: true
        schema:
          $ref: '#/definitions/MetricQuerySchema'
    responses:
      200:
        description: Metric value.
        name: metric
        type: object
        schema:
          $ref: '#/definitions/Metric'
    """
    request_message = _get_request_message(GetMetric())
    response_message = GetMetric.Response()
    metric = _get_store().get_metric(request_message.run_uuid, request_message.metric_key)
    response_message.metric.MergeFrom(metric.to_proto())
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _get_param():
    """
    Get a parameter value.
    ---
    parameters:
      - name: parameter
        description: Parameter identifier.
        in: body
        required: true
        schema:
          $ref: '#/definitions/ParamQuerySchema'
    responses:
      200:
        description: Parameter value.
        name: parameter
        type: object
        schema:
          $ref: '#/definitions/Param'
    """
    request_message = _get_request_message(GetParam())
    response_message = GetParam.Response()
    parameter = _get_store().get_param(request_message.run_uuid, request_message.param_name)
    response_message.parameter.MergeFrom(parameter.to_proto())
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _list_experiments():
    """
    Return a list of all experiments.

    ---
    responses:
      200:
        description: All experiments
        name: experiments
        type: array
        schema:
          $ref: '#/definitions/Experiment'
    """
    response_message = ListExperiments.Response()
    experiment_entities = _get_store().list_experiments()
    response_message.experiments.extend([e.to_proto() for e in experiment_entities])
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _get_artifact_repo(run):
    """
    Get artifact.

    ---
    parameters:
      - $ref: '#/definitions/RunUUIDQuerySchema'
      - name: path
        description: The relative_path to the output base directory.
        in: body
        required: true
        type: string
    responses:
      200:
        description: Artifacts array
        name: artifacts
        schema:
          $ref: '#/definitions/Artifacts
    """
    store = _get_store()
    if run.info.artifact_uri:
        return ArtifactRepository.from_artifact_uri(run.info.artifact_uri, store)

    # TODO(aaron) Remove this once everyone locally only has runs from after
    # the introduction of "artifact_uri".
    uri = os.path.join(store.root_directory, str(run.info.experiment_id),
                       run.info.run_uuid, "artifacts")
    return ArtifactRepository.from_artifact_uri(uri, store)


def _get_paths(base_path):
    """
    A service endpoints base path is typically something like /preview/mlflow/experiment.
    We should register paths like /api/2.0/preview/mlflow/experiment and
    /ajax-api/2.0/preview/mlflow/experiment in the Flask router.
    """
    return ['/api/2.0{}'.format(base_path), '/ajax-api/2.0{}'.format(base_path)]


def get_endpoints():
    """
    :return: List of tuples (path, handler, methods)
    """
    service_methods = MlflowService.DESCRIPTOR.methods
    ret = []
    for service_method in service_methods:
        endpoints = service_method.GetOptions().Extensions[databricks_pb2.rpc].endpoints
        for endpoint in endpoints:
            for http_path in _get_paths(endpoint.path):
                handler = get_handler(MlflowService().GetRequestClass(service_method))
                ret.append((http_path, handler, [endpoint.method]))
    return ret


HANDLERS = {
    CreateExperiment: _create_experiment,
    GetExperiment: _get_experiment,
    CreateRun: _create_run,
    UpdateRun: _update_run,
    LogParam: _log_param,
    LogMetric: _log_metric,
    GetRun: _get_run,
    SearchRuns: _search_runs,
    ListArtifacts: _list_artifacts,
    GetMetricHistory: _get_metric_history,
    ListExperiments: _list_experiments,
    GetParam: _get_param,
    GetMetric: _get_metric,
}
