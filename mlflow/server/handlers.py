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
    UpdateRun, LogMetric, LogParam, SetTag, ListExperiments, GetMetric, GetParam
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
    Returns the ID of the newly created experiment. Validates that another experiment with
    the same name does not already exist and fails if another experiment with the same name
    already exists.
    Throws RESOURCE_ALREADY_EXISTS if a experiment with the given name exists.
    ---
    parameters:
      - name: experiment
        in: body
        required: true
        schema:
          $ref: '#/definitions/ExperimentBody'
    responses:
      200:
        description: Unique identifier for created experiment.
        schema:
          $ref: '#/definitions/ExperimentId'
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
    Get metadata for an experiment and a list of runs for the experiment.
    ---
    parameters:
      - name: experiment_id
        description: Identifier to get an experiment. This field is required.
        in: query
        required: true
        type: integer
        format: int64
    responses:
      200:
        description: Experiment details.
        schema:
          $ref: '#/definitions/ExperimentDetails'
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
          $ref: '#/definitions/RunInfoBody'
    responses:
      200:
        description: Metadata of the newly created run.
        schema:
          $ref: '#/definitions/Run'
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


def _update_run():
    """
    Update run.
    ---
    parameters:
      - name: run_to_update
        description: Run uuid and the status to be updated.
        in: body
        schema:
          $ref: '#/definitions/UpdateRunBody'
    responses:
      200:
        description: Updated metadata of the run.
        schema:
          $ref: '#/definitions/RunInfo2'
    """
    request_message = _get_request_message(UpdateRun())
    updated_info = _get_store().update_run_info(request_message.run_uuid, request_message.status,
                                                request_message.end_time)
    response_message = UpdateRun.Response(run_info=updated_info.to_proto())
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _log_metric():
    """
    Log a metric for a run (e.g. ML model accuracy).
    A metric is a key-value pair (string key, float value) with an associated timestamp.
    Within a run, a metric may be logged multiple times.
    ---
    parameters:
      - name: metric
        description: Metric metadata to be logged.
        in: body
        required: true
        schema:
          $ref: '#/definitions/MetricBody'
    responses:
      200:
        description: OK
    """
    request_message = _get_request_message(LogMetric())
    metric = Metric(request_message.key, request_message.value, request_message.timestamp)
    _get_store().log_metric(request_message.run_uuid, metric)
    response_message = LogMetric.Response()
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _log_param():
    """
    Log a param used for this run.
    Examples are hyperparameters used for ML model training, or constant dates and values used
    in an ETL pipeline. A param is a key-value pair (string key, string value). A param may
    only be logged once for a given run.
    ---
    parameters:
      - name: parameter
        description: Parameter metadata to be logged.
        in: body
        required: true
        schema:
          $ref: '#/definitions/ParamBody'
    responses:
      200:
        description: OK.
    """
    request_message = _get_request_message(LogParam())
    param = Param(request_message.key, request_message.value)
    _get_store().log_param(request_message.run_uuid, param)
    response_message = LogParam.Response()
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _set_tag():
    request_message = _get_request_message(SetTag())
    tag = RunTag(request_message.key, request_message.value)
    _get_store().set_tag(request_message.run_uuid, tag)
    response_message = SetTag.Response()
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
        description: ID of the run to fetch. This field is required.
        in: query
        required: true
        type: string
    responses:
      200:
        description: Run details.
        schema:
          $ref: '#/definitions/RunInfoAndData'
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
      - name: search_body
        description: Request body with search expressions.
        in: body
        schema:
          type: object
          required:
            - experiment_ids
          properties:
            exeriment_ids:
              description: List of experiment IDs to search over.
              type: array
              items:
                type: integer
                format: int64
            anded_expressions:
              description: Expressions describing runs (AND-ed together when filtering runs).
              type: array
              items:
                anyOf:
                  - $ref: '#/definitions/ParameterSearchExpression'
                  - $ref: '#/definitions/MetricSearchExpression'
    responses:
      200:
        description: Runs that match the search criteria.
        schema:
          $ref: '#/definitions/Runs'
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
    List artifacts for a given run.
    Takes an optional artifact_path prefix - if specified, the response will contain only
    artifacts with the specified prefix..
    ---
    parameters:
      - name: run_uuid
        description: ID of the run whose artifacts to list.
        in: query
        required: true
        type: string
      - name: path
        description: Filter artifacts matching this path (a relative path from the root
                     artifact directory).
        in: query
        required: true
        type: string
    responses:
      200:
        description: Artifacts array
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
    Returns a list of all values for the specified metric for a given run.
    ---
    parameters:
      - name: run_uuid
        description: ID of the run from which to fetch metric values. This field is required.
        in: query
        required: true
        type: string
      - name: metric_key
        description: Name of the metric. This field is required.
        in: query
        required: true
        type: string
    responses:
      200:
        description: All logged values for this metric.
        schema:
          $ref: '#/definitions/MetricHistory'
    """
    request_message = _get_request_message(GetMetricHistory())
    response_message = GetMetricHistory.Response()
    metric_entites = _get_store().get_metric_history(request_message.run_uuid,
                                                     request_message.metric_key)
    response_message.metrics.extend([m.to_proto() for m in metric_entites])
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _get_metric():
    """
    Retrieve the logged value for a metric during a run.
    For a run, if this metric is logged more than once, this API retrieves only
    the latest value logged.
    ---
    parameters:
      - name: run_uuid
        description: ID of the run from which to retrieve the metric value. This field is required.
        in: query
        required: true
        type: string
      - name: metric_key
        description: Name of the metric. This field is required.
        in: query
        required: true
        type: string
    responses:
      200:
        description: Latest reported value of the specified metric.
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
    Get a param value.
    ---
    parameters:
      - name: run_uuid
        description: ID of the run from which to retrieve the param value. This field is required.
        in: query
        required: true
        type: string
      - name: param_name
        description: Name of the param. This field is required.
        in: query
        required: true
        type: string
    responses:
      200:
        description: Param key-value pair.
        schema:
          $ref: '#/definitions/Parameter'
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
    Get a list of all experiments.
    ---
    responses:
      200:
        description: All experiments
        schema:
          $ref: '#/definitions/Experiments'
    """
    response_message = ListExperiments.Response()
    experiment_entities = _get_store().list_experiments()
    response_message.experiments.extend([e.to_proto() for e in experiment_entities])
    response = Response(mimetype='application/json')
    response.set_data(_message_to_json(response_message))
    return response


def _get_artifact_repo(run):
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
    SetTag: _set_tag,
    GetRun: _get_run,
    SearchRuns: _search_runs,
    ListArtifacts: _list_artifacts,
    GetMetricHistory: _get_metric_history,
    ListExperiments: _list_experiments,
    GetParam: _get_param,
    GetMetric: _get_metric,
}
