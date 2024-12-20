from mlflow.server.handlers import _get_rest_path

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
