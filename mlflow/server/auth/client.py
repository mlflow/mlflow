from mlflow.server.auth.entities import User, ExperimentPermission, RegisteredModelPermission
from mlflow.server.auth.routes import (
    CREATE_USER,
    GET_USER,
    UPDATE_USER_PASSWORD,
    UPDATE_USER_ADMIN,
    DELETE_USER,
    CREATE_EXPERIMENT_PERMISSION,
    GET_EXPERIMENT_PERMISSION,
    UPDATE_EXPERIMENT_PERMISSION,
    DELETE_EXPERIMENT_PERMISSION,
    CREATE_REGISTERED_MODEL_PERMISSION,
    GET_REGISTERED_MODEL_PERMISSION,
    UPDATE_REGISTERED_MODEL_PERMISSION,
    DELETE_REGISTERED_MODEL_PERMISSION,
)
from mlflow.tracking._tracking_service.utils import _get_default_host_creds
from mlflow.utils.rest_utils import http_request_safe


class AuthServiceClient:
    """
    Client of an MLflow Tracking Server that enabled the default auth plugin.
    """

    def __init__(self, tracking_uri: str):
        """
        :param tracking_uri: Address of local or remote tracking server.
        """
        self.tracking_uri = tracking_uri

    def _request(self, endpoint, method, **kwargs):
        host_creds = _get_default_host_creds(self.tracking_uri)
        resp = http_request_safe(host_creds, endpoint, method, **kwargs)
        return resp.json()

    def create_user(self, username: str, password: str):
        resp = self._request(
            CREATE_USER,
            "POST",
            json={"username": username, "password": password},
        )
        return User.from_json(resp["user"])

    def get_user(self, username: str):
        resp = self._request(
            GET_USER,
            "GET",
            params={"username": username},
        )
        return User.from_json(resp["user"])

    def update_user_password(self, username: str, password: str):
        self._request(
            UPDATE_USER_PASSWORD,
            "PATCH",
            json={"username": username, "password": password},
        )

    def update_user_admin(self, username: str, is_admin: bool):
        self._request(
            UPDATE_USER_ADMIN,
            "PATCH",
            json={"username": username, "is_admin": str(is_admin).lower()},
        )

    def delete_user(self, username: str):
        self._request(
            DELETE_USER,
            "DELETE",
            json={"username": username},
        )

    def create_experiment_permission(self, experiment_id: str, username: str, permission: str):
        resp = self._request(
            CREATE_EXPERIMENT_PERMISSION,
            "POST",
            json={"experiment_id": experiment_id, "username": username, "permission": permission},
        )
        return ExperimentPermission.from_json(resp["experiment_permission"])

    def get_experiment_permission(self, experiment_id: str, username: str):
        resp = self._request(
            GET_EXPERIMENT_PERMISSION,
            "GET",
            params={"experiment_id": experiment_id, "username": username},
        )
        return ExperimentPermission.from_json(resp["experiment_permission"])

    def update_experiment_permission(self, experiment_id: str, username: str, permission: str):
        self._request(
            UPDATE_EXPERIMENT_PERMISSION,
            "PATCH",
            json={"experiment_id": experiment_id, "username": username, "permission": permission},
        )

    def delete_experiment_permission(self, experiment_id: str, username: str):
        self._request(
            DELETE_EXPERIMENT_PERMISSION,
            "DELETE",
            json={"experiment_id": experiment_id, "username": username},
        )

    def create_registered_model_permission(self, name: str, username: str, permission: str):
        resp = self._request(
            CREATE_REGISTERED_MODEL_PERMISSION,
            "POST",
            json={"name": name, "username": username, "permission": permission},
        )
        return RegisteredModelPermission.from_json(resp["registered_model_permission"])

    def get_registered_model_permission(self, name: str, username: str):
        resp = self._request(
            GET_REGISTERED_MODEL_PERMISSION,
            "GET",
            params={"name": name, "username": username},
        )
        return RegisteredModelPermission.from_json(resp["registered_model_permission"])

    def update_registered_model_permission(self, name: str, username: str, permission: str):
        self._request(
            UPDATE_REGISTERED_MODEL_PERMISSION,
            "PATCH",
            json={"name": name, "username": username, "permission": permission},
        )

    def delete_registered_model_permission(self, name: str, username: str):
        self._request(
            DELETE_REGISTERED_MODEL_PERMISSION,
            "DELETE",
            json={"name": name, "username": username},
        )
