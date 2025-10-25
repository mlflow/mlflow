from mlflow.server.handlers import _add_static_prefix, _get_ajax_path, _get_rest_path

HOME = "/"
SIGNUP = "/signup"
CREATE_USER = _get_rest_path("/mlflow/users/create")
CREATE_USER_UI = _get_rest_path("/mlflow/users/create-ui")
GET_USER = _get_rest_path("/mlflow/users/get")
UPDATE_USER_PASSWORD = _get_rest_path("/mlflow/users/update-password")
UPDATE_USER_ADMIN = _get_rest_path("/mlflow/users/update-admin")
DELETE_USER = _get_rest_path("/mlflow/users/delete")
CREATE_EXPERIMENT_PERMISSION = _get_rest_path("/mlflow/experiments/permissions/create")
GET_EXPERIMENT_PERMISSION = _get_rest_path("/mlflow/experiments/permissions/get")
UPDATE_EXPERIMENT_PERMISSION = _get_rest_path("/mlflow/experiments/permissions/update")
DELETE_EXPERIMENT_PERMISSION = _get_rest_path("/mlflow/experiments/permissions/delete")
CREATE_REGISTERED_MODEL_PERMISSION = _get_rest_path("/mlflow/registered-models/permissions/create")
GET_REGISTERED_MODEL_PERMISSION = _get_rest_path("/mlflow/registered-models/permissions/get")
UPDATE_REGISTERED_MODEL_PERMISSION = _get_rest_path("/mlflow/registered-models/permissions/update")
DELETE_REGISTERED_MODEL_PERMISSION = _get_rest_path("/mlflow/registered-models/permissions/delete")

# Flask routes (not part of Protobuf API)
GET_ARTIFACT = _add_static_prefix("/get-artifact")
UPLOAD_ARTIFACT = _get_ajax_path("/mlflow/upload-artifact")
GET_MODEL_VERSION_ARTIFACT = _add_static_prefix("/model-versions/get-artifact")
GET_TRACE_ARTIFACT = _get_ajax_path("/mlflow/get-trace-artifact")
GET_METRIC_HISTORY_BULK = _get_ajax_path("/mlflow/metrics/get-history-bulk")
GET_METRIC_HISTORY_BULK_INTERVAL = _get_ajax_path("/mlflow/metrics/get-history-bulk-interval")
SEARCH_DATASETS = _get_ajax_path("/mlflow/experiments/search-datasets")
CREATE_PROMPTLAB_RUN = _get_ajax_path("/mlflow/runs/create-promptlab-run")
GATEWAY_PROXY = _get_ajax_path("/mlflow/gateway-proxy")
