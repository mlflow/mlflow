from mlflow.server.auth.entities import ExperimentPermission, RegisteredModelPermission
from mlflow.server.auth.routes import (
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
from mlflow.utils.rest_utils import http_request, verify_rest_response


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
        resp = http_request(host_creds, endpoint, method, **kwargs)
        resp = verify_rest_response(resp, endpoint)
        return resp.json()

    def create_experiment_permission(self, experiment_id: str, username: str, permission: str):
        resp = self._request(
            CREATE_EXPERIMENT_PERMISSION,
            "POST",
            json={"experiment_id": experiment_id, "username": username, "permission": permission},
        )
        return ExperimentPermission(
            experiment_id=resp["experiment_permission"]["experiment_id"],
            user_id=resp["experiment_permission"]["user_id"],
            permission=resp["experiment_permission"]["permission"],
        )

    def get_experiment_permission(self, experiment_id: str, username: str):
        resp = self._request(
            GET_EXPERIMENT_PERMISSION,
            "GET",
            params={"experiment_id": experiment_id, "username": username},
        )
        return ExperimentPermission(
            experiment_id=resp["experiment_permission"]["experiment_id"],
            user_id=resp["experiment_permission"]["user_id"],
            permission=resp["experiment_permission"]["permission"],
        )

    def update_experiment_permission(self, experiment_id: str, username: str, permission: str):
        self._request(
            UPDATE_EXPERIMENT_PERMISSION,
            "POST",
            json={"experiment_id": experiment_id, "username": username, "permission": permission},
        )

    def delete_experiment_permission(self, experiment_id: str, username: str):
        self._request(
            DELETE_EXPERIMENT_PERMISSION,
            "POST",
            json={"experiment_id": experiment_id, "username": username},
        )

    def create_registered_model_permission(self, name: str, username: str, permission: str):
        resp = self._request(
            CREATE_REGISTERED_MODEL_PERMISSION,
            "POST",
            json={"name": name, "username": username, "permission": permission},
        )
        return RegisteredModelPermission(
            name=resp["registered_model_permission"]["name"],
            user_id=resp["registered_model_permission"]["user_id"],
            permission=resp["registered_model_permission"]["permission"],
        )

    def get_registered_model_permission(self, name: str, username: str):
        resp = self._request(
            GET_REGISTERED_MODEL_PERMISSION,
            "GET",
            params={"name": name, "username": username},
        )
        return RegisteredModelPermission(
            name=resp["registered_model_permission"]["name"],
            user_id=resp["registered_model_permission"]["user_id"],
            permission=resp["registered_model_permission"]["permission"],
        )

    def update_registered_model_permission(self, name: str, username: str, permission: str):
        self._request(
            UPDATE_REGISTERED_MODEL_PERMISSION,
            "POST",
            json={"name": name, "username": username, "permission": permission},
        )

    def delete_registered_model_permission(self, name: str, username: str):
        self._request(
            DELETE_REGISTERED_MODEL_PERMISSION,
            "POST",
            json={"name": name, "username": username},
        )
