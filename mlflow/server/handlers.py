# Define all the service endpoint handlers here.
import json
import os
import re
import six

from functools import wraps
from flask import Response, request, send_file
from querystring_parser import parser

from mlflow.entities import Metric, Param, RunTag, ViewType, ExperimentTag
from mlflow.exceptions import MlflowException
from mlflow.protos import databricks_pb2
from mlflow.protos.service_pb2 import CreateExperiment, MlflowService, GetExperiment, \
    GetRun, SearchRuns, ListArtifacts, GetMetricHistory, CreateRun, \
    UpdateRun, LogMetric, LogParam, SetTag, ListExperiments, \
    DeleteExperiment, RestoreExperiment, RestoreRun, DeleteRun, UpdateExperiment, LogBatch, \
    DeleteTag, SetExperimentTag, GetExperimentByName
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact_repository_registry import get_artifact_repository
from mlflow.store.dbmodels.db_types import DATABASE_ENGINES
from mlflow.tracking.registry import TrackingStoreRegistry
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.validation import _validate_batch_log_api_req

_store = None
STATIC_PREFIX_ENV_VAR = "_MLFLOW_STATIC_PREFIX"


def _add_static_prefix(route):
    prefix = os.environ.get(STATIC_PREFIX_ENV_VAR)
    if prefix:
        return prefix + route
    return route


def _get_file_store(store_uri, artifact_uri):
    from mlflow.store.file_store import FileStore
    return FileStore(store_uri, artifact_uri)


def _get_sqlalchemy_store(store_uri, artifact_uri):
    from mlflow.store.sqlalchemy_store import SqlAlchemyStore
    return SqlAlchemyStore(store_uri, artifact_uri)


_tracking_store_registry = TrackingStoreRegistry()
_tracking_store_registry.register('', _get_file_store)
_tracking_store_registry.register('file', _get_file_store)
for scheme in DATABASE_ENGINES:
    _tracking_store_registry.register(scheme, _get_sqlalchemy_store)

_tracking_store_registry.register_entrypoints()


def _get_store(backend_store_uri=None, default_artifact_root=None):
    from mlflow.server import BACKEND_STORE_URI_ENV_VAR, ARTIFACT_ROOT_ENV_VAR
    global _store
    if _store is None:
        store_uri = backend_store_uri or os.environ.get(BACKEND_STORE_URI_ENV_VAR, None)
        artifact_root = default_artifact_root or os.environ.get(ARTIFACT_ROOT_ENV_VAR, None)
        _store = _tracking_store_registry.get_store(store_uri, artifact_root)
    return _store


def _get_request_json(flask_request=request):
    return flask_request.get_json(force=True, silent=True)


def _get_request_message(request_message, flask_request=request):
    if flask_request.method == 'GET' and len(flask_request.query_string) > 0:
        # This is a hack to make arrays of length 1 work with the parser.
        # for example experiment_ids%5B%5D=0 should be parsed to {experiment_ids: [0]}
        # but it gets parsed to {experiment_ids: 0}
        # but it doesn't. However, experiment_ids%5B0%5D=0 will get parsed to the right
        # result.
        query_string = re.sub('%5B%5D', '%5B0%5D', flask_request.query_string.decode("utf-8"))
        request_dict = parser.parse(query_string, normalized=True)
        parse_dict(request_dict, request_message)
        return request_message

    request_json = _get_request_json(flask_request)

    # Older clients may post their JSON double-encoded as strings, so the get_json
    # above actually converts it to a string. Therefore, we check this condition
    # (which we can tell for sure because any proper request should be a dictionary),
    # and decode it a second time.
    if isinstance(request_json, six.string_types):
        request_json = json.loads(request_json)

    # If request doesn't have json body then assume it's empty.
    if request_json is None:
        request_json = {}
    parse_dict(request_json, request_message)
    return request_message


def catch_mlflow_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MlflowException as e:
            response = Response(mimetype='application/json')
            response.set_data(e.serialize_as_json())
            response.status_code = e.get_http_status_code()
            return response
    return wrapper


def get_handler(request_class):
    """
    :param request_class: The type of protobuf message
    :return:
    """
    return HANDLERS.get(request_class, _not_implemented)


_TEXT_EXTENSIONS = ['txt', 'log', 'yaml', 'yml', 'json', 'js', 'py',
                    'csv', 'tsv', 'md', 'rst', 'MLmodel', 'MLproject']


@catch_mlflow_exception
def get_artifact_handler():
    query_string = request.query_string.decode('utf-8')
    request_dict = parser.parse(query_string, normalized=True)
    run_id = request_dict.get('run_id') or request_dict.get('run_uuid')
    run = _get_store().get_run(run_id)
    filename = os.path.abspath(_get_artifact_repo(run).download_artifacts(request_dict['path']))
    extension = os.path.splitext(filename)[-1].replace(".", "")
    # Always send artifacts as attachments to prevent the browser from displaying them on our web
    # server's domain, which might enable XSS.
    if extension in _TEXT_EXTENSIONS:
        return send_file(filename, mimetype='text/plain', as_attachment=True)
    else:
        return send_file(filename, as_attachment=True)


def _not_implemented():
    response = Response()
    response.status_code = 404
    return response


@catch_mlflow_exception
def _create_experiment():
    request_message = _get_request_message(CreateExperiment())
    experiment_id = _get_store().create_experiment(request_message.name,
                                                   request_message.artifact_location)
    response_message = CreateExperiment.Response()
    response_message.experiment_id = experiment_id
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _get_experiment():
    request_message = _get_request_message(GetExperiment())
    response_message = GetExperiment.Response()
    experiment = _get_store().get_experiment(request_message.experiment_id).to_proto()
    response_message.experiment.MergeFrom(experiment)
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _get_experiment_by_name():
    request_message = _get_request_message(GetExperimentByName())
    response_message = GetExperimentByName.Response()
    store_exp = _get_store().get_experiment_by_name(request_message.experiment_name)
    if store_exp is None:
        raise MlflowException(
            "Could not find experiment with name '%s'" % request_message.experiment_name,
            error_code=RESOURCE_DOES_NOT_EXIST)
    experiment = store_exp.to_proto()
    response_message.experiment.MergeFrom(experiment)
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _delete_experiment():
    request_message = _get_request_message(DeleteExperiment())
    _get_store().delete_experiment(request_message.experiment_id)
    response_message = DeleteExperiment.Response()
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _restore_experiment():
    request_message = _get_request_message(RestoreExperiment())
    _get_store().restore_experiment(request_message.experiment_id)
    response_message = RestoreExperiment.Response()
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _update_experiment():
    request_message = _get_request_message(UpdateExperiment())
    if request_message.new_name:
        _get_store().rename_experiment(request_message.experiment_id, request_message.new_name)
    response_message = UpdateExperiment.Response()
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _create_run():
    request_message = _get_request_message(CreateRun())

    tags = [RunTag(tag.key, tag.value) for tag in request_message.tags]
    run = _get_store().create_run(
        experiment_id=request_message.experiment_id,
        user_id=request_message.user_id,
        start_time=request_message.start_time,
        tags=tags)

    response_message = CreateRun.Response()
    response_message.run.MergeFrom(run.to_proto())
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _update_run():
    request_message = _get_request_message(UpdateRun())
    run_id = request_message.run_id or request_message.run_uuid
    updated_info = _get_store().update_run_info(run_id, request_message.status,
                                                request_message.end_time)
    response_message = UpdateRun.Response(run_info=updated_info.to_proto())
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _delete_run():
    request_message = _get_request_message(DeleteRun())
    _get_store().delete_run(request_message.run_id)
    response_message = DeleteRun.Response()
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _restore_run():
    request_message = _get_request_message(RestoreRun())
    _get_store().restore_run(request_message.run_id)
    response_message = RestoreRun.Response()
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _log_metric():
    request_message = _get_request_message(LogMetric())
    metric = Metric(request_message.key, request_message.value, request_message.timestamp,
                    request_message.step)
    run_id = request_message.run_id or request_message.run_uuid
    _get_store().log_metric(run_id, metric)
    response_message = LogMetric.Response()
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _log_param():
    request_message = _get_request_message(LogParam())
    param = Param(request_message.key, request_message.value)
    run_id = request_message.run_id or request_message.run_uuid
    _get_store().log_param(run_id, param)
    response_message = LogParam.Response()
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _set_experiment_tag():
    request_message = _get_request_message(SetExperimentTag())
    tag = ExperimentTag(request_message.key, request_message.value)
    _get_store().set_experiment_tag(request_message.experiment_id, tag)
    response_message = SetExperimentTag.Response()
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _set_tag():
    request_message = _get_request_message(SetTag())
    tag = RunTag(request_message.key, request_message.value)
    run_id = request_message.run_id or request_message.run_uuid
    _get_store().set_tag(run_id, tag)
    response_message = SetTag.Response()
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _delete_tag():
    request_message = _get_request_message(DeleteTag())
    _get_store().delete_tag(request_message.run_id, request_message.key)
    response_message = DeleteTag.Response()
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _get_run():
    request_message = _get_request_message(GetRun())
    response_message = GetRun.Response()
    run_id = request_message.run_id or request_message.run_uuid
    response_message.run.MergeFrom(_get_store().get_run(run_id).to_proto())
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _search_runs():
    request_message = _get_request_message(SearchRuns())
    response_message = SearchRuns.Response()
    run_view_type = ViewType.ACTIVE_ONLY
    if request_message.HasField('run_view_type'):
        run_view_type = ViewType.from_proto(request_message.run_view_type)
    filter_string = request_message.filter
    max_results = request_message.max_results
    experiment_ids = request_message.experiment_ids
    order_by = request_message.order_by
    page_token = request_message.page_token
    run_entities = _get_store().search_runs(experiment_ids, filter_string, run_view_type,
                                            max_results, order_by, page_token)
    response_message.runs.extend([r.to_proto() for r in run_entities])
    if run_entities.token:
        response_message.next_page_token = run_entities.token
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _list_artifacts():
    request_message = _get_request_message(ListArtifacts())
    response_message = ListArtifacts.Response()
    if request_message.HasField('path'):
        path = request_message.path
    else:
        path = None
    run_id = request_message.run_id or request_message.run_uuid
    run = _get_store().get_run(run_id)
    artifact_entities = _get_artifact_repo(run).list_artifacts(path)
    response_message.files.extend([a.to_proto() for a in artifact_entities])
    response_message.root_uri = _get_artifact_repo(run).artifact_uri
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _get_metric_history():
    request_message = _get_request_message(GetMetricHistory())
    response_message = GetMetricHistory.Response()
    run_id = request_message.run_id or request_message.run_uuid
    metric_entites = _get_store().get_metric_history(run_id,
                                                     request_message.metric_key)
    response_message.metrics.extend([m.to_proto() for m in metric_entites])
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _list_experiments():
    request_message = _get_request_message(ListExperiments())
    experiment_entities = _get_store().list_experiments(request_message.view_type)
    response_message = ListExperiments.Response()
    response_message.experiments.extend([e.to_proto() for e in experiment_entities])
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _get_artifact_repo(run):
    return get_artifact_repository(run.info.artifact_uri)


@catch_mlflow_exception
def _log_batch():
    _validate_batch_log_api_req(_get_request_json())
    request_message = _get_request_message(LogBatch())
    metrics = [Metric.from_proto(proto_metric) for proto_metric in request_message.metrics]
    params = [Param.from_proto(proto_param) for proto_param in request_message.params]
    tags = [RunTag.from_proto(proto_tag) for proto_tag in request_message.tags]
    _get_store().log_batch(run_id=request_message.run_id, metrics=metrics, params=params, tags=tags)
    response_message = LogBatch.Response()
    response = Response(mimetype='application/json')
    response.set_data(message_to_json(response_message))
    return response


def _get_paths(base_path):
    """
    A service endpoints base path is typically something like /preview/mlflow/experiment.
    We should register paths like /api/2.0/preview/mlflow/experiment and
    /ajax-api/2.0/preview/mlflow/experiment in the Flask router.
    """
    return ['/api/2.0{}'.format(base_path), _add_static_prefix('/ajax-api/2.0{}'.format(base_path))]


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
    GetExperimentByName: _get_experiment_by_name,
    DeleteExperiment: _delete_experiment,
    RestoreExperiment: _restore_experiment,
    UpdateExperiment: _update_experiment,
    CreateRun: _create_run,
    UpdateRun: _update_run,
    DeleteRun: _delete_run,
    RestoreRun: _restore_run,
    LogParam: _log_param,
    LogMetric: _log_metric,
    SetExperimentTag: _set_experiment_tag,
    SetTag: _set_tag,
    DeleteTag: _delete_tag,
    LogBatch: _log_batch,
    GetRun: _get_run,
    SearchRuns: _search_runs,
    ListArtifacts: _list_artifacts,
    GetMetricHistory: _get_metric_history,
    ListExperiments: _list_experiments,
}
