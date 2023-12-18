# Define all the service endpoint handlers here.
import json
import logging
import os
import pathlib
import posixpath
import re
import tempfile
import time
import urllib
from functools import wraps

import requests
from flask import Response, current_app, request, send_file
from google.protobuf import descriptor
from google.protobuf.json_format import ParseError

from mlflow.entities import DatasetInput, ExperimentTag, FileInfo, Metric, Param, RunTag, ViewType
from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.entities.multipart_upload import MultipartUploadPart
from mlflow.environment_variables import (
    MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE,
    MLFLOW_DEPLOYMENTS_TARGET,
)
from mlflow.exceptions import MlflowException, _UnsupportedMultipartUploadException
from mlflow.models import Model
from mlflow.protos import databricks_pb2
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.protos.mlflow_artifacts_pb2 import (
    AbortMultipartUpload,
    CompleteMultipartUpload,
    CreateMultipartUpload,
    DeleteArtifact,
    DownloadArtifact,
    MlflowArtifactsService,
    UploadArtifact,
)
from mlflow.protos.mlflow_artifacts_pb2 import (
    ListArtifacts as ListArtifactsMlflowArtifacts,
)
from mlflow.protos.model_registry_pb2 import (
    CreateModelVersion,
    CreateRegisteredModel,
    DeleteModelVersion,
    DeleteModelVersionTag,
    DeleteRegisteredModel,
    DeleteRegisteredModelAlias,
    DeleteRegisteredModelTag,
    GetLatestVersions,
    GetModelVersion,
    GetModelVersionByAlias,
    GetModelVersionDownloadUri,
    GetRegisteredModel,
    ModelRegistryService,
    RenameRegisteredModel,
    SearchModelVersions,
    SearchRegisteredModels,
    SetModelVersionTag,
    SetRegisteredModelAlias,
    SetRegisteredModelTag,
    TransitionModelVersionStage,
    UpdateModelVersion,
    UpdateRegisteredModel,
)
from mlflow.protos.service_pb2 import (
    CreateExperiment,
    CreateRun,
    DeleteExperiment,
    DeleteRun,
    DeleteTag,
    GetExperiment,
    GetExperimentByName,
    GetMetricHistory,
    GetRun,
    ListArtifacts,
    LogBatch,
    LogInputs,
    LogMetric,
    LogModel,
    LogParam,
    MlflowService,
    RestoreExperiment,
    RestoreRun,
    SearchExperiments,
    SearchRuns,
    SetExperimentTag,
    SetTag,
    UpdateExperiment,
    UpdateRun,
)
from mlflow.server.validation import _validate_content_type
from mlflow.store.artifact.artifact_repo import MultipartUploadMixin
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.tracking._model_registry import utils as registry_utils
from mlflow.tracking._model_registry.registry import ModelRegistryStoreRegistry
from mlflow.tracking._tracking_service import utils
from mlflow.tracking._tracking_service.registry import TrackingStoreRegistry
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.mime_type_utils import _guess_mime_type
from mlflow.utils.promptlab_utils import _create_promptlab_run_impl
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.uri import is_file_uri, is_local_uri, validate_path_is_safe, validate_query_string
from mlflow.utils.validation import _validate_batch_log_api_req

_logger = logging.getLogger(__name__)
_tracking_store = None
_model_registry_store = None
_artifact_repo = None
STATIC_PREFIX_ENV_VAR = "_MLFLOW_STATIC_PREFIX"


class TrackingStoreRegistryWrapper(TrackingStoreRegistry):
    def __init__(self):
        super().__init__()
        self.register("", self._get_file_store)
        self.register("file", self._get_file_store)
        for scheme in DATABASE_ENGINES:
            self.register(scheme, self._get_sqlalchemy_store)
        self.register_entrypoints()

    @classmethod
    def _get_file_store(cls, store_uri, artifact_uri):
        from mlflow.store.tracking.file_store import FileStore

        return FileStore(store_uri, artifact_uri)

    @classmethod
    def _get_sqlalchemy_store(cls, store_uri, artifact_uri):
        from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

        return SqlAlchemyStore(store_uri, artifact_uri)


class ModelRegistryStoreRegistryWrapper(ModelRegistryStoreRegistry):
    def __init__(self):
        super().__init__()
        self.register("", self._get_file_store)
        self.register("file", self._get_file_store)
        for scheme in DATABASE_ENGINES:
            self.register(scheme, self._get_sqlalchemy_store)
        self.register_entrypoints()

    @classmethod
    def _get_file_store(cls, store_uri):
        from mlflow.store.model_registry.file_store import FileStore

        return FileStore(store_uri)

    @classmethod
    def _get_sqlalchemy_store(cls, store_uri):
        from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore

        return SqlAlchemyStore(store_uri)


_tracking_store_registry = TrackingStoreRegistryWrapper()
_model_registry_store_registry = ModelRegistryStoreRegistryWrapper()


def _get_artifact_repo_mlflow_artifacts():
    """
    Get an artifact repository specified by ``--artifacts-destination`` option for ``mlflow server``
    command.
    """
    from mlflow.server import ARTIFACTS_DESTINATION_ENV_VAR

    global _artifact_repo
    if _artifact_repo is None:
        _artifact_repo = get_artifact_repository(os.environ[ARTIFACTS_DESTINATION_ENV_VAR])
    return _artifact_repo


def _is_serving_proxied_artifacts():
    """
    :return: ``True`` if the MLflow server is serving proxied artifacts (i.e. acting as a proxy for
             artifact upload / download / list operations), as would be enabled by specifying the
             ``--serve-artifacts`` configuration option. ``False`` otherwise.
    """
    from mlflow.server import SERVE_ARTIFACTS_ENV_VAR

    return os.environ.get(SERVE_ARTIFACTS_ENV_VAR, "false") == "true"


def _is_servable_proxied_run_artifact_root(run_artifact_root):
    """
    Determines whether or not the following are true:

    - The specified Run artifact root is a proxied artifact root (i.e. an artifact root with scheme
      ``http``, ``https``, or ``mlflow-artifacts``).

    - The MLflow server is capable of resolving and accessing the underlying storage location
      corresponding to the proxied artifact root, allowing it to fulfill artifact list and
      download requests by using this storage location directly.

    :param run_artifact_root: The Run artifact root location (URI).
    :return: ``True`` if the specified Run artifact root refers to proxied artifacts that can be
             served by this MLflow server (i.e. the server has access to the destination and
             can respond to list and download requests for the artifact). ``False`` otherwise.
    """
    parsed_run_artifact_root = urllib.parse.urlparse(run_artifact_root)
    # NB: If the run artifact root is a proxied artifact root (has scheme `http`, `https`, or
    # `mlflow-artifacts`) *and* the MLflow server is configured to serve artifacts, the MLflow
    # server always assumes that it has access to the underlying storage location for the proxied
    # artifacts. This may not always be accurate. For example:
    #
    # An organization may initially use the MLflow server to serve Tracking API requests and proxy
    # access to artifacts stored in Location A (via `mlflow server --serve-artifacts`). Then, for
    # scalability and / or security purposes, the organization may decide to store artifacts in a
    # new location B and set up a separate server (e.g. `mlflow server --artifacts-only`) to proxy
    # access to artifacts stored in Location B.
    #
    # In this scenario, requests for artifacts stored in Location B that are sent to the original
    # MLflow server will fail if the original MLflow server does not have access to Location B
    # because it will assume that it can serve all proxied artifacts regardless of the underlying
    # location. Such failures can be remediated by granting the original MLflow server access to
    # Location B.
    return (
        parsed_run_artifact_root.scheme in ["http", "https", "mlflow-artifacts"]
        and _is_serving_proxied_artifacts()
    )


def _get_proxied_run_artifact_destination_path(proxied_artifact_root, relative_path=None):
    """
    Resolves the specified proxied artifact location within a Run to a concrete storage location.

    :param proxied_artifact_root: The Run artifact root location (URI) with scheme ``http``,
                                  ``https``, or `mlflow-artifacts` that can be resolved by the
                                  MLflow server to a concrete storage location.
    :param relative_path: The relative path of the destination within the specified
                          ``proxied_artifact_root``. If ``None``, the destination is assumed to be
                          the resolved ``proxied_artifact_root``.
    :return: The storage location of the specified artifact.
    """
    parsed_proxied_artifact_root = urllib.parse.urlparse(proxied_artifact_root)
    assert parsed_proxied_artifact_root.scheme in ["http", "https", "mlflow-artifacts"]

    if parsed_proxied_artifact_root.scheme == "mlflow-artifacts":
        # If the proxied artifact root is an `mlflow-artifacts` URI, the run artifact root path is
        # simply the path component of the URI, since the fully-qualified format of an
        # `mlflow-artifacts` URI is `mlflow-artifacts://<netloc>/path/to/artifact`
        proxied_run_artifact_root_path = parsed_proxied_artifact_root.path.lstrip("/")
    else:
        # In this case, the proxied artifact root is an HTTP(S) URL referring to an mlflow-artifacts
        # API route that can be used to download the artifact. These routes are always anchored at
        # `/api/2.0/mlflow-artifacts/artifacts`. Accordingly, we split the path on this route anchor
        # and interpret the rest of the path (everything after the route anchor) as the run artifact
        # root path
        mlflow_artifacts_http_route_anchor = "/api/2.0/mlflow-artifacts/artifacts/"
        assert mlflow_artifacts_http_route_anchor in parsed_proxied_artifact_root.path

        proxied_run_artifact_root_path = parsed_proxied_artifact_root.path.split(
            mlflow_artifacts_http_route_anchor
        )[1].lstrip("/")

    return (
        posixpath.join(proxied_run_artifact_root_path, relative_path)
        if relative_path is not None
        else proxied_run_artifact_root_path
    )


def _get_tracking_store(backend_store_uri=None, default_artifact_root=None):
    from mlflow.server import ARTIFACT_ROOT_ENV_VAR, BACKEND_STORE_URI_ENV_VAR

    global _tracking_store
    if _tracking_store is None:
        store_uri = backend_store_uri or os.environ.get(BACKEND_STORE_URI_ENV_VAR, None)
        artifact_root = default_artifact_root or os.environ.get(ARTIFACT_ROOT_ENV_VAR, None)
        _tracking_store = _tracking_store_registry.get_store(store_uri, artifact_root)
        utils.set_tracking_uri(store_uri)
    return _tracking_store


def _get_model_registry_store(registry_store_uri=None):
    from mlflow.server import BACKEND_STORE_URI_ENV_VAR, REGISTRY_STORE_URI_ENV_VAR

    global _model_registry_store
    if _model_registry_store is None:
        store_uri = (
            registry_store_uri
            or os.environ.get(REGISTRY_STORE_URI_ENV_VAR, None)
            or os.environ.get(BACKEND_STORE_URI_ENV_VAR, None)
        )
        _model_registry_store = _model_registry_store_registry.get_store(store_uri)
        registry_utils.set_registry_uri(store_uri)
    return _model_registry_store


def initialize_backend_stores(
    backend_store_uri=None, registry_store_uri=None, default_artifact_root=None
):
    _get_tracking_store(backend_store_uri, default_artifact_root)
    try:
        _get_model_registry_store(registry_store_uri)
    except UnsupportedModelRegistryStoreURIException:
        pass


def _assert_string(x):
    assert isinstance(x, str)


def _assert_intlike(x):
    try:
        x = int(x)
    except ValueError:
        pass

    assert isinstance(x, int)


def _assert_bool(x):
    assert isinstance(x, bool)


def _assert_floatlike(x):
    try:
        x = float(x)
    except ValueError:
        pass

    assert isinstance(x, float)


def _assert_array(x):
    assert isinstance(x, list)


def _assert_required(x):
    assert x is not None
    # When parsing JSON payloads via proto, absent string fields
    # are expressed as empty strings
    assert x != ""


def _assert_less_than_or_equal(x, max_value):
    assert x <= max_value


def _assert_item_type_string(x):
    assert all(isinstance(item, str) for item in x)


_TYPE_VALIDATORS = {
    _assert_intlike,
    _assert_string,
    _assert_bool,
    _assert_floatlike,
    _assert_array,
    _assert_item_type_string,
}


def _validate_param_against_schema(schema, param, value, proto_parsing_succeeded=False):
    """
    Attempts to validate a single parameter against a specified schema.
    Examples of the elements of the schema are type assertions and checks for required parameters.
    Returns None on validation success. Otherwise, raises an MLFlowException if an assertion fails.
    This method is intended to be called for side effects.

            Parameters:
    :param schema: A list of functions to validate the parameter against.
    :param param: The string name of the parameter being validated.
    :param value: The corresponding value of the `param` being validated.
    :param proto_parsing_succeeded: A boolean value indicating whether proto parsing succeeded.
                                    If the proto was successfully parsed, we assume all of the types
                                    of the parameters in the request body were correctly specified,
                                    and thus we skip validating types. If proto parsing failed,
                                    then we validate types in addition to the rest of the schema.
                                    For details, see https://github.com/mlflow/mlflow/pull/
                                    5458#issuecomment-1080880870.
    """

    for f in schema:
        if f in _TYPE_VALIDATORS and proto_parsing_succeeded:
            continue

        try:
            f(value)
        except AssertionError:
            if f == _assert_required:
                message = f"Missing value for required parameter '{param}'."
            else:
                message = (
                    f"Invalid value {value} for parameter '{param}' supplied."
                    f" Hint: Value was of type '{type(value).__name__}'."
                )
            raise MlflowException(
                message=(
                    message + " See the API docs for more information about request parameters."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

    return None


def _get_request_json(flask_request=request):
    _validate_content_type(flask_request, ["application/json"])
    return flask_request.get_json(force=True, silent=True)


def _get_request_message(request_message, flask_request=request, schema=None):
    from querystring_parser import parser

    if flask_request.method == "GET" and len(flask_request.query_string) > 0:
        # This is a hack to make arrays of length 1 work with the parser.
        # for example experiment_ids%5B%5D=0 should be parsed to {experiment_ids: [0]}
        # but it gets parsed to {experiment_ids: 0}
        # but it doesn't. However, experiment_ids%5B0%5D=0 will get parsed to the right
        # result.
        query_string = re.sub("%5B%5D", "%5B0%5D", flask_request.query_string.decode("utf-8"))
        request_dict = parser.parse(query_string, normalized=True)
        # Convert atomic values of repeated fields to lists before calling protobuf deserialization.
        # Context: We parse the parameter string into a dictionary outside of protobuf since
        # protobuf does not know how to read the query parameters directly. The query parser above
        # has no type information and hence any parameter that occurs exactly once is parsed as an
        # atomic value. Since protobuf requires that the values of repeated fields are lists,
        # deserialization will fail unless we do the fix below.
        for field in request_message.DESCRIPTOR.fields:
            if (
                field.label == descriptor.FieldDescriptor.LABEL_REPEATED
                and field.name in request_dict
            ):
                if not isinstance(request_dict[field.name], list):
                    request_dict[field.name] = [request_dict[field.name]]
        parse_dict(request_dict, request_message)
        return request_message

    request_json = _get_request_json(flask_request)

    # Older clients may post their JSON double-encoded as strings, so the get_json
    # above actually converts it to a string. Therefore, we check this condition
    # (which we can tell for sure because any proper request should be a dictionary),
    # and decode it a second time.
    if is_string_type(request_json):
        request_json = json.loads(request_json)

    # If request doesn't have json body then assume it's empty.
    if request_json is None:
        request_json = {}

    proto_parsing_succeeded = True
    try:
        parse_dict(request_json, request_message)
    except ParseError:
        proto_parsing_succeeded = False

    schema = schema or {}
    for schema_key, schema_validation_fns in schema.items():
        if schema_key in request_json or _assert_required in schema_validation_fns:
            value = request_json.get(schema_key)
            if schema_key == "run_id" and value is None and "run_uuid" in request_json:
                value = request_json.get("run_uuid")
            _validate_param_against_schema(
                schema=schema_validation_fns,
                param=schema_key,
                value=value,
                proto_parsing_succeeded=proto_parsing_succeeded,
            )

    return request_message


def _response_with_file_attachment_headers(file_path, response):
    mime_type = _guess_mime_type(file_path)
    filename = pathlib.Path(file_path).name
    response.mimetype = mime_type
    content_disposition_header_name = "Content-Disposition"
    if content_disposition_header_name not in response.headers:
        response.headers[content_disposition_header_name] = f"attachment; filename={filename}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Type"] = mime_type
    return response


def _send_artifact(artifact_repository, path):
    file_path = os.path.abspath(artifact_repository.download_artifacts(path))
    # Always send artifacts as attachments to prevent the browser from displaying them on our web
    # server's domain, which might enable XSS.
    mime_type = _guess_mime_type(file_path)
    file_sender_response = send_file(file_path, mimetype=mime_type, as_attachment=True)
    return _response_with_file_attachment_headers(file_path, file_sender_response)


def catch_mlflow_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MlflowException as e:
            response = Response(mimetype="application/json")
            response.set_data(e.serialize_as_json())
            response.status_code = e.get_http_status_code()
            return response

    return wrapper


def _disable_unless_serve_artifacts(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _is_serving_proxied_artifacts():
            return Response(
                (
                    f"Endpoint: {request.url_rule} disabled due to the mlflow server running "
                    "with `--no-serve-artifacts`. To enable artifacts server functionality, "
                    "run `mlflow server` with `--serve-artifacts`"
                ),
                503,
            )
        return func(*args, **kwargs)

    return wrapper


def _disable_if_artifacts_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        from mlflow.server import ARTIFACTS_ONLY_ENV_VAR

        if os.environ.get(ARTIFACTS_ONLY_ENV_VAR):
            return Response(
                (
                    f"Endpoint: {request.url_rule} disabled due to the mlflow server running "
                    "in `--artifacts-only` mode. To enable tracking server functionality, run "
                    "`mlflow server` without `--artifacts-only`"
                ),
                503,
            )
        return func(*args, **kwargs)

    return wrapper


@catch_mlflow_exception
def get_artifact_handler():
    from querystring_parser import parser

    query_string = request.query_string.decode("utf-8")
    request_dict = parser.parse(query_string, normalized=True)
    run_id = request_dict.get("run_id") or request_dict.get("run_uuid")
    path = request_dict["path"]
    path = validate_path_is_safe(path)
    run = _get_tracking_store().get_run(run_id)

    if _is_servable_proxied_run_artifact_root(run.info.artifact_uri):
        artifact_repo = _get_artifact_repo_mlflow_artifacts()
        artifact_path = _get_proxied_run_artifact_destination_path(
            proxied_artifact_root=run.info.artifact_uri,
            relative_path=path,
        )
    else:
        artifact_repo = _get_artifact_repo(run)
        artifact_path = path

    return _send_artifact(artifact_repo, artifact_path)


def _not_implemented():
    response = Response()
    response.status_code = 404
    return response


# Tracking Server APIs


@catch_mlflow_exception
@_disable_if_artifacts_only
def _create_experiment():
    request_message = _get_request_message(
        CreateExperiment(),
        schema={
            "name": [_assert_required, _assert_string],
            "artifact_location": [_assert_string],
            "tags": [_assert_array],
        },
    )

    tags = [ExperimentTag(tag.key, tag.value) for tag in request_message.tags]

    # Validate query string in artifact location to prevent attacks
    parsed_artifact_locaion = urllib.parse.urlparse(request_message.artifact_location)
    validate_query_string(parsed_artifact_locaion.query)

    experiment_id = _get_tracking_store().create_experiment(
        request_message.name, request_message.artifact_location, tags
    )
    response_message = CreateExperiment.Response()
    response_message.experiment_id = experiment_id
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_experiment():
    request_message = _get_request_message(
        GetExperiment(), schema={"experiment_id": [_assert_required, _assert_string]}
    )
    response_message = GetExperiment.Response()
    experiment = _get_tracking_store().get_experiment(request_message.experiment_id).to_proto()
    response_message.experiment.MergeFrom(experiment)
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_experiment_by_name():
    request_message = _get_request_message(
        GetExperimentByName(), schema={"experiment_name": [_assert_required, _assert_string]}
    )
    response_message = GetExperimentByName.Response()
    store_exp = _get_tracking_store().get_experiment_by_name(request_message.experiment_name)
    if store_exp is None:
        raise MlflowException(
            f"Could not find experiment with name '{request_message.experiment_name}'",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )
    experiment = store_exp.to_proto()
    response_message.experiment.MergeFrom(experiment)
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _delete_experiment():
    request_message = _get_request_message(
        DeleteExperiment(), schema={"experiment_id": [_assert_required, _assert_string]}
    )
    _get_tracking_store().delete_experiment(request_message.experiment_id)
    response_message = DeleteExperiment.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _restore_experiment():
    request_message = _get_request_message(
        RestoreExperiment(), schema={"experiment_id": [_assert_required, _assert_string]}
    )
    _get_tracking_store().restore_experiment(request_message.experiment_id)
    response_message = RestoreExperiment.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _update_experiment():
    request_message = _get_request_message(
        UpdateExperiment(),
        schema={
            "experiment_id": [_assert_required, _assert_string],
            "new_name": [_assert_string, _assert_required],
        },
    )
    if request_message.new_name:
        _get_tracking_store().rename_experiment(
            request_message.experiment_id, request_message.new_name
        )
    response_message = UpdateExperiment.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _create_run():
    request_message = _get_request_message(
        CreateRun(),
        schema={
            "experiment_id": [_assert_string],
            "start_time": [_assert_intlike],
            "run_name": [_assert_string],
        },
    )

    tags = [RunTag(tag.key, tag.value) for tag in request_message.tags]
    run = _get_tracking_store().create_run(
        experiment_id=request_message.experiment_id,
        user_id=request_message.user_id,
        start_time=request_message.start_time,
        tags=tags,
        run_name=request_message.run_name,
    )

    response_message = CreateRun.Response()
    response_message.run.MergeFrom(run.to_proto())
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _update_run():
    request_message = _get_request_message(
        UpdateRun(),
        schema={
            "run_id": [_assert_required, _assert_string],
            "end_time": [_assert_intlike],
            "status": [_assert_string],
            "run_name": [_assert_string],
        },
    )
    run_id = request_message.run_id or request_message.run_uuid
    run_name = request_message.run_name if request_message.HasField("run_name") else None
    end_time = request_message.end_time if request_message.HasField("end_time") else None
    status = request_message.status if request_message.HasField("status") else None
    updated_info = _get_tracking_store().update_run_info(run_id, status, end_time, run_name)
    response_message = UpdateRun.Response(run_info=updated_info.to_proto())
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _delete_run():
    request_message = _get_request_message(
        DeleteRun(), schema={"run_id": [_assert_required, _assert_string]}
    )
    _get_tracking_store().delete_run(request_message.run_id)
    response_message = DeleteRun.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _restore_run():
    request_message = _get_request_message(
        RestoreRun(), schema={"run_id": [_assert_required, _assert_string]}
    )
    _get_tracking_store().restore_run(request_message.run_id)
    response_message = RestoreRun.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _log_metric():
    request_message = _get_request_message(
        LogMetric(),
        schema={
            "run_id": [_assert_required, _assert_string],
            "key": [_assert_required, _assert_string],
            "value": [_assert_required, _assert_floatlike],
            "timestamp": [_assert_intlike, _assert_required],
            "step": [_assert_intlike],
        },
    )
    metric = Metric(
        request_message.key, request_message.value, request_message.timestamp, request_message.step
    )
    run_id = request_message.run_id or request_message.run_uuid
    _get_tracking_store().log_metric(run_id, metric)
    response_message = LogMetric.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _log_param():
    request_message = _get_request_message(
        LogParam(),
        schema={
            "run_id": [_assert_required, _assert_string],
            "key": [_assert_required, _assert_string],
            "value": [_assert_string],
        },
    )
    param = Param(request_message.key, request_message.value)
    run_id = request_message.run_id or request_message.run_uuid
    _get_tracking_store().log_param(run_id, param)
    response_message = LogParam.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _log_inputs():
    request_message = _get_request_message(
        LogInputs(),
        schema={
            "run_id": [_assert_required, _assert_string],
            "datasets": [_assert_required, _assert_array],
        },
    )
    run_id = request_message.run_id
    datasets = [
        DatasetInput.from_proto(proto_dataset_input)
        for proto_dataset_input in request_message.datasets
    ]

    _get_tracking_store().log_inputs(run_id, datasets=datasets)
    response_message = LogInputs.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _set_experiment_tag():
    request_message = _get_request_message(
        SetExperimentTag(),
        schema={
            "experiment_id": [_assert_required, _assert_string],
            "key": [_assert_required, _assert_string],
            "value": [_assert_string],
        },
    )
    tag = ExperimentTag(request_message.key, request_message.value)
    _get_tracking_store().set_experiment_tag(request_message.experiment_id, tag)
    response_message = SetExperimentTag.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _set_tag():
    request_message = _get_request_message(
        SetTag(),
        schema={
            "run_id": [_assert_required, _assert_string],
            "key": [_assert_required, _assert_string],
            "value": [_assert_string],
        },
    )
    tag = RunTag(request_message.key, request_message.value)
    run_id = request_message.run_id or request_message.run_uuid
    _get_tracking_store().set_tag(run_id, tag)
    response_message = SetTag.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _delete_tag():
    request_message = _get_request_message(
        DeleteTag(),
        schema={
            "run_id": [_assert_required, _assert_string],
            "key": [_assert_required, _assert_string],
        },
    )
    _get_tracking_store().delete_tag(request_message.run_id, request_message.key)
    response_message = DeleteTag.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_run():
    request_message = _get_request_message(
        GetRun(), schema={"run_id": [_assert_required, _assert_string]}
    )
    response_message = GetRun.Response()
    run_id = request_message.run_id or request_message.run_uuid
    response_message.run.MergeFrom(_get_tracking_store().get_run(run_id).to_proto())
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _search_runs():
    request_message = _get_request_message(
        SearchRuns(),
        schema={
            "experiment_ids": [_assert_array],
            "filter": [_assert_string],
            "max_results": [_assert_intlike, lambda x: _assert_less_than_or_equal(x, 50000)],
            "order_by": [_assert_array, _assert_item_type_string],
        },
    )
    response_message = SearchRuns.Response()
    run_view_type = ViewType.ACTIVE_ONLY
    if request_message.HasField("run_view_type"):
        run_view_type = ViewType.from_proto(request_message.run_view_type)
    filter_string = request_message.filter
    max_results = request_message.max_results
    experiment_ids = request_message.experiment_ids
    order_by = request_message.order_by
    page_token = request_message.page_token
    run_entities = _get_tracking_store().search_runs(
        experiment_ids, filter_string, run_view_type, max_results, order_by, page_token
    )
    response_message.runs.extend([r.to_proto() for r in run_entities])
    if run_entities.token:
        response_message.next_page_token = run_entities.token
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _list_artifacts():
    request_message = _get_request_message(
        ListArtifacts(),
        schema={
            "run_id": [_assert_string, _assert_required],
            "path": [_assert_string],
            "page_token": [_assert_string],
        },
    )
    response_message = ListArtifacts.Response()
    if request_message.HasField("path"):
        path = request_message.path
        path = validate_path_is_safe(path)
    else:
        path = None
    run_id = request_message.run_id or request_message.run_uuid
    run = _get_tracking_store().get_run(run_id)

    if _is_servable_proxied_run_artifact_root(run.info.artifact_uri):
        artifact_entities = _list_artifacts_for_proxied_run_artifact_root(
            proxied_artifact_root=run.info.artifact_uri,
            relative_path=path,
        )
    else:
        artifact_entities = _get_artifact_repo(run).list_artifacts(path)

    response_message.files.extend([a.to_proto() for a in artifact_entities])
    response_message.root_uri = run.info.artifact_uri
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _list_artifacts_for_proxied_run_artifact_root(proxied_artifact_root, relative_path=None):
    """
    Lists artifacts from the specified ``relative_path`` within the specified proxied Run artifact
    root (i.e. a Run artifact root with scheme ``http``, ``https``, or ``mlflow-artifacts``).

    :param proxied_artifact_root: The Run artifact root location (URI) with scheme ``http``,
                                  ``https``, or ``mlflow-artifacts`` that can be resolved by the
                                  MLflow server to a concrete storage location.
    :param relative_path: The relative path within the specified ``proxied_artifact_root`` under
                          which to list artifact contents. If ``None``, artifacts are listed from
                          the ``proxied_artifact_root`` directory.
    """
    parsed_proxied_artifact_root = urllib.parse.urlparse(proxied_artifact_root)
    assert parsed_proxied_artifact_root.scheme in ["http", "https", "mlflow-artifacts"]

    artifact_destination_repo = _get_artifact_repo_mlflow_artifacts()
    artifact_destination_path = _get_proxied_run_artifact_destination_path(
        proxied_artifact_root=proxied_artifact_root,
        relative_path=relative_path,
    )

    artifact_entities = []
    for file_info in artifact_destination_repo.list_artifacts(artifact_destination_path):
        basename = posixpath.basename(file_info.path)
        run_relative_artifact_path = (
            posixpath.join(relative_path, basename) if relative_path else basename
        )
        artifact_entities.append(
            FileInfo(run_relative_artifact_path, file_info.is_dir, file_info.file_size)
        )

    return artifact_entities


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_metric_history():
    request_message = _get_request_message(
        GetMetricHistory(),
        schema={
            "run_id": [_assert_string, _assert_required],
            "metric_key": [_assert_string, _assert_required],
        },
    )
    response_message = GetMetricHistory.Response()
    run_id = request_message.run_id or request_message.run_uuid
    metric_entities = _get_tracking_store().get_metric_history(run_id, request_message.metric_key)
    response_message.metrics.extend([m.to_proto() for m in metric_entities])
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def get_metric_history_bulk_handler():
    MAX_HISTORY_RESULTS = 25000
    MAX_RUN_IDS_PER_REQUEST = 20
    run_ids = request.args.to_dict(flat=False).get("run_id", [])
    if not run_ids:
        raise MlflowException(
            message="GetMetricHistoryBulk request must specify at least one run_id.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if len(run_ids) > MAX_RUN_IDS_PER_REQUEST:
        raise MlflowException(
            message=(
                f"GetMetricHistoryBulk request cannot specify more than {MAX_RUN_IDS_PER_REQUEST}"
                f" run_ids. Received {len(run_ids)} run_ids."
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    metric_key = request.args.get("metric_key")
    if metric_key is None:
        raise MlflowException(
            message="GetMetricHistoryBulk request must specify a metric_key.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    max_results = int(request.args.get("max_results", MAX_HISTORY_RESULTS))
    max_results = min(max_results, MAX_HISTORY_RESULTS)

    store = _get_tracking_store()

    def _default_history_bulk_impl():
        metrics_with_run_ids = []
        for run_id in sorted(run_ids):
            metrics_for_run = sorted(
                store.get_metric_history(
                    run_id=run_id,
                    metric_key=metric_key,
                    max_results=max_results,
                ),
                key=lambda metric: (metric.timestamp, metric.step, metric.value),
            )
            metrics_with_run_ids.extend(
                [
                    {
                        "key": metric.key,
                        "value": metric.value,
                        "timestamp": metric.timestamp,
                        "step": metric.step,
                        "run_id": run_id,
                    }
                    for metric in metrics_for_run
                ]
            )
        return metrics_with_run_ids

    if hasattr(store, "get_metric_history_bulk"):
        metrics_with_run_ids = [
            metric.to_dict()
            for metric in store.get_metric_history_bulk(
                run_ids=run_ids,
                metric_key=metric_key,
                max_results=max_results,
            )
        ]
    else:
        metrics_with_run_ids = _default_history_bulk_impl()

    return {
        "metrics": metrics_with_run_ids[:max_results],
    }


@catch_mlflow_exception
@_disable_if_artifacts_only
def search_datasets_handler():
    MAX_EXPERIMENT_IDS_PER_REQUEST = 20
    _validate_content_type(request, ["application/json"])
    experiment_ids = request.json.get("experiment_ids", [])
    if not experiment_ids:
        raise MlflowException(
            message="SearchDatasets request must specify at least one experiment_id.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if len(experiment_ids) > MAX_EXPERIMENT_IDS_PER_REQUEST:
        raise MlflowException(
            message=(
                f"SearchDatasets request cannot specify more than {MAX_EXPERIMENT_IDS_PER_REQUEST}"
                f" experiment_ids. Received {len(experiment_ids)} experiment_ids."
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    store = _get_tracking_store()

    if hasattr(store, "_search_datasets"):
        return {
            "dataset_summaries": [
                summary.to_dict() for summary in store._search_datasets(experiment_ids)
            ]
        }
    else:
        return _not_implemented()


@catch_mlflow_exception
def gateway_proxy_handler():
    target_uri = MLFLOW_DEPLOYMENTS_TARGET.get()
    if not target_uri:
        # Pretend an empty gateway service is running
        return {"endpoints": []}

    args = request.args if request.method == "GET" else request.json

    gateway_path = args.get("gateway_path")
    if not gateway_path:
        raise MlflowException(
            message="Deployments proxy request must specify a gateway_path.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    request_type = request.method
    json_data = args.get("json_data", None)

    response = requests.request(request_type, f"{target_uri}/{gateway_path}", json=json_data)

    if response.status_code == 200:
        return response.json()
    else:
        raise MlflowException(
            message=f"Deployments proxy request failed with error code {response.status_code}. "
            f"Error message: {response.text}",
            error_code=response.status_code,
        )


@catch_mlflow_exception
@_disable_if_artifacts_only
def create_promptlab_run_handler():
    def assert_arg_exists(arg_name, arg):
        if not arg:
            raise MlflowException(
                message=f"CreatePromptlabRun request must specify {arg_name}.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    _validate_content_type(request, ["application/json"])

    args = request.json
    experiment_id = args.get("experiment_id")
    assert_arg_exists("experiment_id", experiment_id)
    run_name = args.get("run_name", None)
    tags = args.get("tags", [])
    prompt_template = args.get("prompt_template")
    assert_arg_exists("prompt_template", prompt_template)
    raw_prompt_parameters = args.get("prompt_parameters")
    assert_arg_exists("prompt_parameters", raw_prompt_parameters)
    prompt_parameters = [
        Param(param.get("key"), param.get("value")) for param in args.get("prompt_parameters")
    ]
    model_route = args.get("model_route")
    assert_arg_exists("model_route", model_route)
    raw_model_parameters = args.get("model_parameters", [])
    model_parameters = [
        Param(param.get("key"), param.get("value")) for param in raw_model_parameters
    ]
    model_input = args.get("model_input")
    assert_arg_exists("model_input", model_input)
    model_output = args.get("model_output", None)
    raw_model_output_parameters = args.get("model_output_parameters", [])
    model_output_parameters = [
        Param(param.get("key"), param.get("value")) for param in raw_model_output_parameters
    ]
    mlflow_version = args.get("mlflow_version")
    assert_arg_exists("mlflow_version", mlflow_version)
    user_id = args.get("user_id", "unknown")

    # use current time if not provided
    start_time = args.get("start_time", int(time.time() * 1000))

    store = _get_tracking_store()

    run = _create_promptlab_run_impl(
        store,
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,
        prompt_template=prompt_template,
        prompt_parameters=prompt_parameters,
        model_route=model_route,
        model_parameters=model_parameters,
        model_input=model_input,
        model_output=model_output,
        model_output_parameters=model_output_parameters,
        mlflow_version=mlflow_version,
        user_id=user_id,
        start_time=start_time,
    )
    response_message = CreateRun.Response()
    response_message.run.MergeFrom(run.to_proto())
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def upload_artifact_handler():
    args = request.args
    run_uuid = args.get("run_uuid")
    if not run_uuid:
        raise MlflowException(
            message="Request must specify run_uuid.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    path = args.get("path")
    if not path:
        raise MlflowException(
            message="Request must specify path.",
            error_code=INVALID_PARAMETER_VALUE,
        )
    path = validate_path_is_safe(path)

    if request.content_length and request.content_length > 10 * 1024 * 1024:
        raise MlflowException(
            message="Artifact size is too large. Max size is 10MB.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    data = request.data
    if not data:
        raise MlflowException(
            message="Request must specify data.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    run = _get_tracking_store().get_run(run_uuid)
    artifact_dir = run.info.artifact_uri

    basename = posixpath.basename(path)
    dirname = posixpath.dirname(path)

    def _log_artifact_to_repo(file, run, dirname, artifact_dir):
        if _is_servable_proxied_run_artifact_root(run.info.artifact_uri):
            artifact_repo = _get_artifact_repo_mlflow_artifacts()
            path_to_log = (
                os.path.join(run.info.experiment_id, run.info.run_id, "artifacts", dirname)
                if dirname
                else os.path.join(run.info.experiment_id, run.info.run_id, "artifacts")
            )
        else:
            artifact_repo = get_artifact_repository(artifact_dir)
            path_to_log = dirname

        artifact_repo.log_artifact(file, path_to_log)

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = os.path.join(tmpdir, dirname) if dirname else tmpdir
        file_path = os.path.join(dir_path, basename)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(data)

        _log_artifact_to_repo(file_path, run, dirname, artifact_dir)

    return Response(mimetype="application/json")


@catch_mlflow_exception
@_disable_if_artifacts_only
def _search_experiments():
    request_message = _get_request_message(
        SearchExperiments(),
        schema={
            "view_type": [_assert_intlike],
            "max_results": [_assert_intlike],
            "order_by": [_assert_array],
            "filter": [_assert_string],
            "page_token": [_assert_string],
        },
    )
    experiment_entities = _get_tracking_store().search_experiments(
        view_type=request_message.view_type,
        max_results=request_message.max_results,
        order_by=request_message.order_by,
        filter_string=request_message.filter,
        page_token=request_message.page_token,
    )
    response_message = SearchExperiments.Response()
    response_message.experiments.extend([e.to_proto() for e in experiment_entities])
    if experiment_entities.token:
        response_message.next_page_token = experiment_entities.token
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
def _get_artifact_repo(run):
    return get_artifact_repository(run.info.artifact_uri)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _log_batch():
    def _assert_metrics_fields_present(metrics):
        for m in metrics:
            _assert_required(m.get("key"))
            _assert_required(m.get("value"))
            _assert_required(m.get("timestamp"))

    def _assert_params_tags_fields_present(params_or_tags):
        for param_or_tag in params_or_tags:
            _assert_required(param_or_tag.get("key"))

    _validate_batch_log_api_req(_get_request_json())
    request_message = _get_request_message(
        LogBatch(),
        schema={
            "run_id": [_assert_string, _assert_required],
            "metrics": [_assert_array, _assert_metrics_fields_present],
            "params": [_assert_array, _assert_params_tags_fields_present],
            "tags": [_assert_array, _assert_params_tags_fields_present],
        },
    )
    metrics = [Metric.from_proto(proto_metric) for proto_metric in request_message.metrics]
    params = [Param.from_proto(proto_param) for proto_param in request_message.params]
    tags = [RunTag.from_proto(proto_tag) for proto_tag in request_message.tags]
    _get_tracking_store().log_batch(
        run_id=request_message.run_id, metrics=metrics, params=params, tags=tags
    )
    response_message = LogBatch.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_if_artifacts_only
def _log_model():
    request_message = _get_request_message(
        LogModel(),
        schema={
            "run_id": [_assert_string, _assert_required],
            "model_json": [_assert_string, _assert_required],
        },
    )
    try:
        model = json.loads(request_message.model_json)
    except Exception:
        raise MlflowException(
            f"Malformed model info. \n {request_message.model_json} \n is not a valid JSON.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    missing_fields = {"artifact_path", "flavors", "utc_time_created", "run_id"} - set(model.keys())

    if missing_fields:
        raise MlflowException(
            f"Model json is missing mandatory fields: {missing_fields}",
            error_code=INVALID_PARAMETER_VALUE,
        )
    _get_tracking_store().record_logged_model(
        run_id=request_message.run_id, mlflow_model=Model.from_dict(model)
    )
    response_message = LogModel.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


def _wrap_response(response_message):
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


# Model Registry APIs


@catch_mlflow_exception
@_disable_if_artifacts_only
def _create_registered_model():
    request_message = _get_request_message(
        CreateRegisteredModel(),
        schema={
            "name": [_assert_string, _assert_required],
            "tags": [_assert_array],
            "description": [_assert_string],
        },
    )
    registered_model = _get_model_registry_store().create_registered_model(
        name=request_message.name,
        tags=request_message.tags,
        description=request_message.description,
    )
    response_message = CreateRegisteredModel.Response(registered_model=registered_model.to_proto())
    return _wrap_response(response_message)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_registered_model():
    request_message = _get_request_message(
        GetRegisteredModel(), schema={"name": [_assert_string, _assert_required]}
    )
    registered_model = _get_model_registry_store().get_registered_model(name=request_message.name)
    response_message = GetRegisteredModel.Response(registered_model=registered_model.to_proto())
    return _wrap_response(response_message)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _update_registered_model():
    request_message = _get_request_message(
        UpdateRegisteredModel(),
        schema={"name": [_assert_string, _assert_required], "description": [_assert_string]},
    )
    name = request_message.name
    new_description = request_message.description
    registered_model = _get_model_registry_store().update_registered_model(
        name=name, description=new_description
    )
    response_message = UpdateRegisteredModel.Response(registered_model=registered_model.to_proto())
    return _wrap_response(response_message)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _rename_registered_model():
    request_message = _get_request_message(
        RenameRegisteredModel(),
        schema={
            "name": [_assert_string, _assert_required],
            "new_name": [_assert_string, _assert_required],
        },
    )
    name = request_message.name
    new_name = request_message.new_name
    registered_model = _get_model_registry_store().rename_registered_model(
        name=name, new_name=new_name
    )
    response_message = RenameRegisteredModel.Response(registered_model=registered_model.to_proto())
    return _wrap_response(response_message)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _delete_registered_model():
    request_message = _get_request_message(
        DeleteRegisteredModel(), schema={"name": [_assert_string, _assert_required]}
    )
    _get_model_registry_store().delete_registered_model(name=request_message.name)
    return _wrap_response(DeleteRegisteredModel.Response())


@catch_mlflow_exception
@_disable_if_artifacts_only
def _search_registered_models():
    request_message = _get_request_message(
        SearchRegisteredModels(),
        schema={
            "filter": [_assert_string],
            "max_results": [_assert_intlike, lambda x: _assert_less_than_or_equal(x, 1000)],
            "order_by": [_assert_array, _assert_item_type_string],
            "page_token": [_assert_string],
        },
    )
    store = _get_model_registry_store()
    registered_models = store.search_registered_models(
        filter_string=request_message.filter,
        max_results=request_message.max_results,
        order_by=request_message.order_by,
        page_token=request_message.page_token,
    )
    response_message = SearchRegisteredModels.Response()
    response_message.registered_models.extend([e.to_proto() for e in registered_models])
    if registered_models.token:
        response_message.next_page_token = registered_models.token
    return _wrap_response(response_message)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_latest_versions():
    request_message = _get_request_message(
        GetLatestVersions(),
        schema={
            "name": [_assert_string, _assert_required],
            "stages": [_assert_array, _assert_item_type_string],
        },
    )
    latest_versions = _get_model_registry_store().get_latest_versions(
        name=request_message.name, stages=request_message.stages
    )
    response_message = GetLatestVersions.Response()
    response_message.model_versions.extend([e.to_proto() for e in latest_versions])
    return _wrap_response(response_message)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _set_registered_model_tag():
    request_message = _get_request_message(
        SetRegisteredModelTag(),
        schema={
            "name": [_assert_string, _assert_required],
            "key": [_assert_string, _assert_required],
            "value": [_assert_string],
        },
    )
    tag = RegisteredModelTag(key=request_message.key, value=request_message.value)
    _get_model_registry_store().set_registered_model_tag(name=request_message.name, tag=tag)
    return _wrap_response(SetRegisteredModelTag.Response())


@catch_mlflow_exception
@_disable_if_artifacts_only
def _delete_registered_model_tag():
    request_message = _get_request_message(
        DeleteRegisteredModelTag(),
        schema={
            "name": [_assert_string, _assert_required],
            "key": [_assert_string, _assert_required],
        },
    )
    _get_model_registry_store().delete_registered_model_tag(
        name=request_message.name, key=request_message.key
    )
    return _wrap_response(DeleteRegisteredModelTag.Response())


def _validate_non_local_source_contains_relative_paths(source: str):
    """
    Validation check to ensure that sources that are provided that conform to the schemes:
    http, https, or mlflow-artifacts do not contain relative path designations that are intended
    to access local file system paths on the tracking server.

    Example paths that this validation function is intended to find and raise an Exception if
    passed:
    "mlflow-artifacts://host:port/../../../../"
    "http://host:port/api/2.0/mlflow-artifacts/artifacts/../../../../"
    "https://host:port/api/2.0/mlflow-artifacts/artifacts/../../../../"
    "/models/artifacts/../../../"
    "s3:/my_bucket/models/path/../../other/path"
    "file://path/to/../../../../some/where/you/should/not/be"
    "mlflow-artifacts://host:port/..%2f..%2f..%2f..%2f"
    "http://host:port/api/2.0/mlflow-artifacts/artifacts%00"
    """
    invalid_source_error_message = (
        f"Invalid model version source: '{source}'. If supplying a source as an http, https, "
        "local file path, ftp, objectstore, or mlflow-artifacts uri, an absolute path must be "
        "provided without relative path references present. "
        "Please provide an absolute path."
    )

    while (unquoted := urllib.parse.unquote_plus(source)) != source:
        source = unquoted
    source_path = re.sub(r"/+", "/", urllib.parse.urlparse(source).path.rstrip("/"))
    if "\x00" in source_path:
        raise MlflowException(invalid_source_error_message, INVALID_PARAMETER_VALUE)
    resolved_source = pathlib.Path(source_path).resolve().as_posix()
    # NB: drive split is specifically for Windows since WindowsPath.resolve() will append the
    # drive path of the pwd to a given path. We don't care about the drive here, though.
    _, resolved_path = os.path.splitdrive(resolved_source)

    if resolved_path != source_path:
        raise MlflowException(invalid_source_error_message, INVALID_PARAMETER_VALUE)


def _validate_source(source: str, run_id: str) -> None:
    if is_local_uri(source):
        if run_id:
            store = _get_tracking_store()
            run = store.get_run(run_id)
            source = pathlib.Path(local_file_uri_to_path(source)).resolve()
            run_artifact_dir = pathlib.Path(local_file_uri_to_path(run.info.artifact_uri)).resolve()
            if run_artifact_dir in [source, *source.parents]:
                return

        raise MlflowException(
            f"Invalid model version source: '{source}'. To use a local path as a model version "
            "source, the run_id request parameter has to be specified and the local path has to be "
            "contained within the artifact directory of the run specified by the run_id.",
            INVALID_PARAMETER_VALUE,
        )

    # There might be file URIs that are local but can bypass the above check. To prevent this, we
    # disallow using file URIs as model version sources by default unless it's explicitly allowed
    # by setting the MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE environment variable to True.
    if not MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE.get() and is_file_uri(source):
        raise MlflowException(
            f"Invalid model version source: '{source}'. MLflow tracking server doesn't allow using "
            "a file URI as a model version source for security reasons. To disable this check, set "
            f"the {MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE} environment variable to "
            "True.",
            INVALID_PARAMETER_VALUE,
        )

    # Checks if relative paths are present in the source (a security threat). If any are present,
    # raises an Exception.
    _validate_non_local_source_contains_relative_paths(source)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _create_model_version():
    request_message = _get_request_message(
        CreateModelVersion(),
        schema={
            "name": [_assert_string, _assert_required],
            "source": [_assert_string, _assert_required],
            "run_id": [_assert_string],
            "tags": [_assert_array],
            "run_link": [_assert_string],
            "description": [_assert_string],
        },
    )

    _validate_source(request_message.source, request_message.run_id)

    model_version = _get_model_registry_store().create_model_version(
        name=request_message.name,
        source=request_message.source,
        run_id=request_message.run_id,
        run_link=request_message.run_link,
        tags=request_message.tags,
        description=request_message.description,
    )
    response_message = CreateModelVersion.Response(model_version=model_version.to_proto())
    return _wrap_response(response_message)


@catch_mlflow_exception
@_disable_if_artifacts_only
def get_model_version_artifact_handler():
    from querystring_parser import parser

    query_string = request.query_string.decode("utf-8")
    request_dict = parser.parse(query_string, normalized=True)
    name = request_dict.get("name")
    version = request_dict.get("version")
    path = request_dict["path"]
    path = validate_path_is_safe(path)
    artifact_uri = _get_model_registry_store().get_model_version_download_uri(name, version)
    if _is_servable_proxied_run_artifact_root(artifact_uri):
        artifact_repo = _get_artifact_repo_mlflow_artifacts()
        artifact_path = _get_proxied_run_artifact_destination_path(
            proxied_artifact_root=artifact_uri,
            relative_path=path,
        )
    else:
        artifact_repo = get_artifact_repository(artifact_uri)
        artifact_path = path

    return _send_artifact(artifact_repo, artifact_path)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_model_version():
    request_message = _get_request_message(
        GetModelVersion(),
        schema={
            "name": [_assert_string, _assert_required],
            "version": [_assert_string, _assert_required],
        },
    )
    model_version = _get_model_registry_store().get_model_version(
        name=request_message.name, version=request_message.version
    )
    response_proto = model_version.to_proto()
    response_message = GetModelVersion.Response(model_version=response_proto)
    return _wrap_response(response_message)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _update_model_version():
    request_message = _get_request_message(
        UpdateModelVersion(),
        schema={
            "name": [_assert_string, _assert_required],
            "version": [_assert_string, _assert_required],
            "description": [_assert_string],
        },
    )
    new_description = None
    if request_message.HasField("description"):
        new_description = request_message.description
    model_version = _get_model_registry_store().update_model_version(
        name=request_message.name, version=request_message.version, description=new_description
    )
    return _wrap_response(UpdateModelVersion.Response(model_version=model_version.to_proto()))


@catch_mlflow_exception
@_disable_if_artifacts_only
def _transition_stage():
    request_message = _get_request_message(
        TransitionModelVersionStage(),
        schema={
            "name": [_assert_string, _assert_required],
            "version": [_assert_string, _assert_required],
            "stage": [_assert_string, _assert_required],
            "archive_existing_versions": [_assert_bool],
        },
    )
    model_version = _get_model_registry_store().transition_model_version_stage(
        name=request_message.name,
        version=request_message.version,
        stage=request_message.stage,
        archive_existing_versions=request_message.archive_existing_versions,
    )
    return _wrap_response(
        TransitionModelVersionStage.Response(model_version=model_version.to_proto())
    )


@catch_mlflow_exception
@_disable_if_artifacts_only
def _delete_model_version():
    request_message = _get_request_message(
        DeleteModelVersion(),
        schema={
            "name": [_assert_string, _assert_required],
            "version": [_assert_string, _assert_required],
        },
    )
    _get_model_registry_store().delete_model_version(
        name=request_message.name, version=request_message.version
    )
    return _wrap_response(DeleteModelVersion.Response())


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_model_version_download_uri():
    request_message = _get_request_message(GetModelVersionDownloadUri())
    download_uri = _get_model_registry_store().get_model_version_download_uri(
        name=request_message.name, version=request_message.version
    )
    response_message = GetModelVersionDownloadUri.Response(artifact_uri=download_uri)
    return _wrap_response(response_message)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _search_model_versions():
    request_message = _get_request_message(
        SearchModelVersions(),
        schema={
            "filter": [_assert_string],
            "max_results": [_assert_intlike, lambda x: _assert_less_than_or_equal(x, 200_000)],
            "order_by": [_assert_array, _assert_item_type_string],
            "page_token": [_assert_string],
        },
    )
    store = _get_model_registry_store()
    model_versions = store.search_model_versions(
        filter_string=request_message.filter,
        max_results=request_message.max_results,
        order_by=request_message.order_by,
        page_token=request_message.page_token,
    )
    response_message = SearchModelVersions.Response()
    response_message.model_versions.extend([e.to_proto() for e in model_versions])
    if model_versions.token:
        response_message.next_page_token = model_versions.token
    return _wrap_response(response_message)


@catch_mlflow_exception
@_disable_if_artifacts_only
def _set_model_version_tag():
    request_message = _get_request_message(
        SetModelVersionTag(),
        schema={
            "name": [_assert_string, _assert_required],
            "version": [_assert_string, _assert_required],
            "key": [_assert_string, _assert_required],
            "value": [_assert_string],
        },
    )
    tag = ModelVersionTag(key=request_message.key, value=request_message.value)
    _get_model_registry_store().set_model_version_tag(
        name=request_message.name, version=request_message.version, tag=tag
    )
    return _wrap_response(SetModelVersionTag.Response())


@catch_mlflow_exception
@_disable_if_artifacts_only
def _delete_model_version_tag():
    request_message = _get_request_message(
        DeleteModelVersionTag(),
        schema={
            "name": [_assert_string, _assert_required],
            "version": [_assert_string, _assert_required],
            "key": [_assert_string, _assert_required],
        },
    )
    _get_model_registry_store().delete_model_version_tag(
        name=request_message.name, version=request_message.version, key=request_message.key
    )
    return _wrap_response(DeleteModelVersionTag.Response())


@catch_mlflow_exception
@_disable_if_artifacts_only
def _set_registered_model_alias():
    request_message = _get_request_message(
        SetRegisteredModelAlias(),
        schema={
            "name": [_assert_string, _assert_required],
            "alias": [_assert_string, _assert_required],
            "version": [_assert_string, _assert_required],
        },
    )
    _get_model_registry_store().set_registered_model_alias(
        name=request_message.name, alias=request_message.alias, version=request_message.version
    )
    return _wrap_response(SetRegisteredModelAlias.Response())


@catch_mlflow_exception
@_disable_if_artifacts_only
def _delete_registered_model_alias():
    request_message = _get_request_message(
        DeleteRegisteredModelAlias(),
        schema={
            "name": [_assert_string, _assert_required],
            "alias": [_assert_string, _assert_required],
        },
    )
    _get_model_registry_store().delete_registered_model_alias(
        name=request_message.name, alias=request_message.alias
    )
    return _wrap_response(DeleteRegisteredModelAlias.Response())


@catch_mlflow_exception
@_disable_if_artifacts_only
def _get_model_version_by_alias():
    request_message = _get_request_message(
        GetModelVersionByAlias(),
        schema={
            "name": [_assert_string, _assert_required],
            "alias": [_assert_string, _assert_required],
        },
    )
    model_version = _get_model_registry_store().get_model_version_by_alias(
        name=request_message.name, alias=request_message.alias
    )
    response_proto = model_version.to_proto()
    response_message = GetModelVersionByAlias.Response(model_version=response_proto)
    return _wrap_response(response_message)


# MLflow Artifacts APIs


@catch_mlflow_exception
@_disable_unless_serve_artifacts
def _download_artifact(artifact_path):
    """
    A request handler for `GET /mlflow-artifacts/artifacts/<artifact_path>` to download an artifact
    from `artifact_path` (a relative path from the root artifact directory).
    """
    artifact_path = validate_path_is_safe(artifact_path)
    tmp_dir = tempfile.TemporaryDirectory()
    artifact_repo = _get_artifact_repo_mlflow_artifacts()
    dst = artifact_repo.download_artifacts(artifact_path, tmp_dir.name)

    # Ref: https://stackoverflow.com/a/24613980/6943581
    file_handle = open(dst, "rb")  # noqa: SIM115

    def stream_and_remove_file():
        yield from file_handle
        file_handle.close()
        tmp_dir.cleanup()

    file_sender_response = current_app.response_class(stream_and_remove_file())

    return _response_with_file_attachment_headers(artifact_path, file_sender_response)


@catch_mlflow_exception
@_disable_unless_serve_artifacts
def _upload_artifact(artifact_path):
    """
    A request handler for `PUT /mlflow-artifacts/artifacts/<artifact_path>` to upload an artifact
    to `artifact_path` (a relative path from the root artifact directory).
    """
    artifact_path = validate_path_is_safe(artifact_path)
    head, tail = posixpath.split(artifact_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, tail)
        with open(tmp_path, "wb") as f:
            chunk_size = 1024 * 1024  # 1 MB
            while True:
                chunk = request.stream.read(chunk_size)
                if len(chunk) == 0:
                    break
                f.write(chunk)

        artifact_repo = _get_artifact_repo_mlflow_artifacts()
        artifact_repo.log_artifact(tmp_path, artifact_path=head or None)

    return _wrap_response(UploadArtifact.Response())


@catch_mlflow_exception
@_disable_unless_serve_artifacts
def _list_artifacts_mlflow_artifacts():
    """
    A request handler for `GET /mlflow-artifacts/artifacts?path=<value>` to list artifacts in `path`
    (a relative path from the root artifact directory).
    """
    request_message = _get_request_message(ListArtifactsMlflowArtifacts())
    path = validate_path_is_safe(request_message.path) if request_message.HasField("path") else None
    artifact_repo = _get_artifact_repo_mlflow_artifacts()
    files = []
    for file_info in artifact_repo.list_artifacts(path):
        basename = posixpath.basename(file_info.path)
        new_file_info = FileInfo(basename, file_info.is_dir, file_info.file_size)
        files.append(new_file_info.to_proto())
    response_message = ListArtifacts.Response()
    response_message.files.extend(files)
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_unless_serve_artifacts
def _delete_artifact_mlflow_artifacts(artifact_path):
    """
    A request handler for `DELETE /mlflow-artifacts/artifacts?path=<value>` to delete artifacts in
    `path` (a relative path from the root artifact directory).
    """
    artifact_path = validate_path_is_safe(artifact_path)
    _get_request_message(DeleteArtifact())
    artifact_repo = _get_artifact_repo_mlflow_artifacts()
    artifact_repo.delete_artifacts(artifact_path)
    response_message = DeleteArtifact.Response()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


def _validate_support_multipart_upload(artifact_repo):
    if not isinstance(artifact_repo, MultipartUploadMixin):
        raise _UnsupportedMultipartUploadException()


@catch_mlflow_exception
@_disable_unless_serve_artifacts
def _create_multipart_upload_artifact(artifact_path):
    """
    A request handler for `POST /mlflow-artifacts/mpu/create` to create a multipart upload
    to `artifact_path` (a relative path from the root artifact directory).
    """
    artifact_path = validate_path_is_safe(artifact_path)

    request_message = _get_request_message(
        CreateMultipartUpload(),
        schema={
            "path": [_assert_required, _assert_string],
            "num_parts": [_assert_intlike],
        },
    )
    path = request_message.path
    num_parts = request_message.num_parts

    artifact_repo = _get_artifact_repo_mlflow_artifacts()
    _validate_support_multipart_upload(artifact_repo)

    create_response = artifact_repo.create_multipart_upload(
        path,
        num_parts,
        artifact_path,
    )
    response_message = create_response.to_proto()
    response = Response(mimetype="application/json")
    response.set_data(message_to_json(response_message))
    return response


@catch_mlflow_exception
@_disable_unless_serve_artifacts
def _complete_multipart_upload_artifact(artifact_path):
    """
    A request handler for `POST /mlflow-artifacts/mpu/complete` to complete a multipart upload
    to `artifact_path` (a relative path from the root artifact directory).
    """
    artifact_path = validate_path_is_safe(artifact_path)

    request_message = _get_request_message(
        CompleteMultipartUpload(),
        schema={
            "path": [_assert_required, _assert_string],
            "upload_id": [_assert_string],
            "parts": [_assert_required],
        },
    )
    path = request_message.path
    upload_id = request_message.upload_id
    parts = [MultipartUploadPart.from_proto(part) for part in request_message.parts]

    artifact_repo = _get_artifact_repo_mlflow_artifacts()
    _validate_support_multipart_upload(artifact_repo)

    artifact_repo.complete_multipart_upload(
        path,
        upload_id,
        parts,
        artifact_path,
    )
    return _wrap_response(CompleteMultipartUpload.Response())


@catch_mlflow_exception
@_disable_unless_serve_artifacts
def _abort_multipart_upload_artifact(artifact_path):
    """
    A request handler for `POST /mlflow-artifacts/mpu/abort` to abort a multipart upload
    to `artifact_path` (a relative path from the root artifact directory).
    """
    artifact_path = validate_path_is_safe(artifact_path)

    request_message = _get_request_message(
        AbortMultipartUpload(),
        schema={
            "path": [_assert_required, _assert_string],
            "upload_id": [_assert_string],
        },
    )
    path = request_message.path
    upload_id = request_message.upload_id

    artifact_repo = _get_artifact_repo_mlflow_artifacts()
    _validate_support_multipart_upload(artifact_repo)

    artifact_repo.abort_multipart_upload(
        path,
        upload_id,
        artifact_path,
    )
    return _wrap_response(AbortMultipartUpload.Response())


def _get_rest_path(base_path):
    return f"/api/2.0{base_path}"


def _get_ajax_path(base_path):
    return _add_static_prefix(f"/ajax-api/2.0{base_path}")


def _add_static_prefix(route):
    prefix = os.environ.get(STATIC_PREFIX_ENV_VAR)
    if prefix:
        return prefix + route
    return route


def _get_paths(base_path):
    """
    A service endpoints base path is typically something like /mlflow/experiment.
    We should register paths like /api/2.0/mlflow/experiment and
    /ajax-api/2.0/mlflow/experiment in the Flask router.
    """
    return [_get_rest_path(base_path), _get_ajax_path(base_path)]


def get_handler(request_class):
    """
    :param request_class: The type of protobuf message
    :return:
    """
    return HANDLERS.get(request_class, _not_implemented)


def get_service_endpoints(service, get_handler):
    ret = []
    for service_method in service.DESCRIPTOR.methods:
        endpoints = service_method.GetOptions().Extensions[databricks_pb2.rpc].endpoints
        for endpoint in endpoints:
            for http_path in _get_paths(endpoint.path):
                handler = get_handler(service().GetRequestClass(service_method))
                ret.append((http_path, handler, [endpoint.method]))
    return ret


def get_endpoints(get_handler=get_handler):
    """
    :return: List of tuples (path, handler, methods)
    """
    return (
        get_service_endpoints(MlflowService, get_handler)
        + get_service_endpoints(ModelRegistryService, get_handler)
        + get_service_endpoints(MlflowArtifactsService, get_handler)
    )


HANDLERS = {
    # Tracking Server APIs
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
    LogModel: _log_model,
    GetRun: _get_run,
    SearchRuns: _search_runs,
    ListArtifacts: _list_artifacts,
    GetMetricHistory: _get_metric_history,
    SearchExperiments: _search_experiments,
    LogInputs: _log_inputs,
    # Model Registry APIs
    CreateRegisteredModel: _create_registered_model,
    GetRegisteredModel: _get_registered_model,
    DeleteRegisteredModel: _delete_registered_model,
    UpdateRegisteredModel: _update_registered_model,
    RenameRegisteredModel: _rename_registered_model,
    SearchRegisteredModels: _search_registered_models,
    GetLatestVersions: _get_latest_versions,
    CreateModelVersion: _create_model_version,
    GetModelVersion: _get_model_version,
    DeleteModelVersion: _delete_model_version,
    UpdateModelVersion: _update_model_version,
    TransitionModelVersionStage: _transition_stage,
    GetModelVersionDownloadUri: _get_model_version_download_uri,
    SearchModelVersions: _search_model_versions,
    SetRegisteredModelTag: _set_registered_model_tag,
    DeleteRegisteredModelTag: _delete_registered_model_tag,
    SetModelVersionTag: _set_model_version_tag,
    DeleteModelVersionTag: _delete_model_version_tag,
    SetRegisteredModelAlias: _set_registered_model_alias,
    DeleteRegisteredModelAlias: _delete_registered_model_alias,
    GetModelVersionByAlias: _get_model_version_by_alias,
    # MLflow Artifacts APIs
    DownloadArtifact: _download_artifact,
    UploadArtifact: _upload_artifact,
    ListArtifactsMlflowArtifacts: _list_artifacts_mlflow_artifacts,
    DeleteArtifact: _delete_artifact_mlflow_artifacts,
    CreateMultipartUpload: _create_multipart_upload_artifact,
    CompleteMultipartUpload: _complete_multipart_upload_artifact,
    AbortMultipartUpload: _abort_multipart_upload_artifact,
}
