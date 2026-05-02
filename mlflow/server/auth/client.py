from urllib.parse import quote

from mlflow.server.auth.entities import (
    ExperimentPermission,
    GatewayEndpointPermission,
    GatewayModelDefinitionPermission,
    GatewaySecretPermission,
    RegisteredModelPermission,
    Role,
    RolePermission,
    ScorerPermission,
    User,
    UserRoleAssignment,
    WorkspacePermission,
)
from mlflow.server.auth.routes import (
    ADD_ROLE_PERMISSION,
    ASSIGN_ROLE,
    CREATE_EXPERIMENT_PERMISSION,
    CREATE_GATEWAY_ENDPOINT_PERMISSION,
    CREATE_GATEWAY_MODEL_DEFINITION_PERMISSION,
    CREATE_GATEWAY_SECRET_PERMISSION,
    CREATE_REGISTERED_MODEL_PERMISSION,
    CREATE_ROLE,
    CREATE_SCORER_PERMISSION,
    CREATE_USER,
    DELETE_EXPERIMENT_PERMISSION,
    DELETE_GATEWAY_ENDPOINT_PERMISSION,
    DELETE_GATEWAY_MODEL_DEFINITION_PERMISSION,
    DELETE_GATEWAY_SECRET_PERMISSION,
    DELETE_REGISTERED_MODEL_PERMISSION,
    DELETE_ROLE,
    DELETE_SCORER_PERMISSION,
    DELETE_USER,
    GET_EXPERIMENT_PERMISSION,
    GET_GATEWAY_ENDPOINT_PERMISSION,
    GET_GATEWAY_MODEL_DEFINITION_PERMISSION,
    GET_GATEWAY_SECRET_PERMISSION,
    GET_REGISTERED_MODEL_PERMISSION,
    GET_ROLE,
    GET_SCORER_PERMISSION,
    GET_USER,
    LIST_ROLE_PERMISSIONS,
    LIST_ROLE_USERS,
    LIST_ROLES,
    LIST_USER_ROLES,
    LIST_USER_WORKSPACE_PERMISSIONS,
    LIST_WORKSPACE_PERMISSIONS,
    REMOVE_ROLE_PERMISSION,
    UNASSIGN_ROLE,
    UPDATE_EXPERIMENT_PERMISSION,
    UPDATE_GATEWAY_ENDPOINT_PERMISSION,
    UPDATE_GATEWAY_MODEL_DEFINITION_PERMISSION,
    UPDATE_GATEWAY_SECRET_PERMISSION,
    UPDATE_REGISTERED_MODEL_PERMISSION,
    UPDATE_ROLE,
    UPDATE_ROLE_PERMISSION,
    UPDATE_SCORER_PERMISSION,
    UPDATE_USER_ADMIN,
    UPDATE_USER_PASSWORD,
)
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.rest_utils import http_request, verify_rest_response


class AuthServiceClient:
    """
    Client of an MLflow Tracking Server that enabled the default basic authentication plugin.
    It is recommended to use :py:func:`mlflow.server.get_app_client()` to instantiate this class.
    See https://mlflow.org/docs/latest/auth.html for more information.
    """

    def __init__(self, tracking_uri: str):
        """
        Args:
            tracking_uri: Address of local or remote tracking server.
        """
        self.tracking_uri = tracking_uri

    def _request(self, endpoint, method, *, expected_status: int = 200, **kwargs):
        host_creds = get_default_host_creds(self.tracking_uri)
        resp = http_request(host_creds, endpoint, method, **kwargs)
        resp = verify_rest_response(resp, endpoint, expected_status=expected_status)
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    def _workspace_endpoint(self, template: str, workspace_name: str) -> str:
        """
        Replace the workspace placeholder in ``template`` with a URL-encoded name.
        """
        return template.replace("<workspace_name>", quote(workspace_name, safe=""))

    def create_user(self, username: str, password: str):
        """
        Create a new user.

        Args:
            username: The username.
            password: The user's password. Must not be empty string.

        Raises:
            mlflow.exceptions.RestException: if the username is already taken.

        Returns:
            A single :py:class:`mlflow.server.auth.entities.User` object.

        .. code-block:: python
            :caption: Example

            from mlflow.server.auth.client import AuthServiceClient

            client = AuthServiceClient("tracking_uri")
            user = client.create_user("newuser", "newpassword")
            print(f"user_id: {user.id}")
            print(f"username: {user.username}")
            print(f"password_hash: {user.password_hash}")
            print(f"is_admin: {user.is_admin}")

        .. code-block:: text
            :caption: Output

            user_id: 3
            username: newuser
            password_hash: REDACTED
            is_admin: False
        """
        resp = self._request(
            CREATE_USER,
            "POST",
            json={"username": username, "password": password},
        )
        return User.from_json(resp["user"])

    def get_user(self, username: str):
        """
        Get a user with a specific username.

        Args:
            username: The username.

        Raises:
            mlflow.exceptions.RestException: if the user does not exist

        Returns:
            A single :py:class:`mlflow.server.auth.entities.User` object.

        .. code-block:: bash
            :caption: Example

            export MLFLOW_TRACKING_USERNAME=admin
            export MLFLOW_TRACKING_PASSWORD=password

        .. code-block:: python

            from mlflow.server.auth.client import AuthServiceClient

            client = AuthServiceClient("tracking_uri")
            client.create_user("newuser", "newpassword")
            user = client.get_user("newuser")

            print(f"user_id: {user.id}")
            print(f"username: {user.username}")
            print(f"password_hash: {user.password_hash}")
            print(f"is_admin: {user.is_admin}")

        .. code-block:: text
            :caption: Output

            user_id: 3
            username: newuser
            password_hash: REDACTED
            is_admin: False
        """
        resp = self._request(
            GET_USER,
            "GET",
            params={"username": username},
        )
        return User.from_json(resp["user"])

    def update_user_password(
        self, username: str, password: str, current_password: str | None = None
    ):
        """
        Update the password of a specific user.

        Args:
            username: The username.
            password: The new password.
            current_password: The user's current password. Required when a user
                is changing their own password (self-service); the server
                rejects the request otherwise. Admins changing someone else's
                password may omit this argument.

        Raises:
            mlflow.exceptions.RestException: if the user does not exist, or if
                ``current_password`` is required and missing or incorrect.

        .. code-block:: bash
            :caption: Example

            export MLFLOW_TRACKING_USERNAME=admin
            export MLFLOW_TRACKING_PASSWORD=password

        .. code-block:: python

            from mlflow.server.auth.client import AuthServiceClient

            client = AuthServiceClient("tracking_uri")
            client.create_user("newuser", "newpassword")

            # Admin path — no current_password needed.
            client.update_user_password("newuser", "anotherpassword")

            # Self-service path — current_password required.
            client.update_user_password(
                "newuser", "thirdpassword", current_password="anotherpassword"
            )
        """
        body = {"username": username, "password": password}
        if current_password is not None:
            body["current_password"] = current_password
        self._request(
            UPDATE_USER_PASSWORD,
            "PATCH",
            json=body,
        )

    def update_user_admin(self, username: str, is_admin: bool):
        """
        Update the admin status of a specific user.

        Args:
            username: The username.
            is_admin: The new admin status.

        Raises:
            mlflow.exceptions.RestException: if the user does not exist

        .. code-block:: bash
            :caption: Example

            export MLFLOW_TRACKING_USERNAME=admin
            export MLFLOW_TRACKING_PASSWORD=password

        .. code-block:: python

            from mlflow.server.auth.client import AuthServiceClient

            client = AuthServiceClient("tracking_uri")
            client.create_user("newuser", "newpassword")

            client.update_user_admin("newuser", True)
        """
        self._request(
            UPDATE_USER_ADMIN,
            "PATCH",
            json={"username": username, "is_admin": is_admin},
        )

    def delete_user(self, username: str):
        """
        Delete a specific user.

        Args:
            username: The username.

        Raises:
            mlflow.exceptions.RestException: if the user does not exist

        .. code-block:: bash
            :caption: Example

            export MLFLOW_TRACKING_USERNAME=admin
            export MLFLOW_TRACKING_PASSWORD=password

        .. code-block:: python

            from mlflow.server.auth.client import AuthServiceClient

            client = AuthServiceClient("tracking_uri")
            client.create_user("newuser", "newpassword")

            client.delete_user("newuser")
        """
        self._request(
            DELETE_USER,
            "DELETE",
            json={"username": username},
        )

    def create_experiment_permission(self, experiment_id: str, username: str, permission: str):
        """
        Create a permission on an experiment for a user.

        Args:
            experiment_id: The id of the experiment.
            username: The username.
            permission: Permission to grant. Must be one of "READ", "USE", "EDIT", "MANAGE" and
                "NO_PERMISSIONS".

        Raises:
            mlflow.exceptions.RestException: if the user does not exist, or a permission already
                exists for this experiment user pair, or if the permission is invalid. Does not
                require ``experiment_id`` to be an existing experiment.

        Returns:
            A single :py:class:`mlflow.server.auth.entities.ExperimentPermission` object.

        .. code-block:: bash
            :caption: Example

            export MLFLOW_TRACKING_USERNAME=admin
            export MLFLOW_TRACKING_PASSWORD=password

        .. code-block:: python

            from mlflow.server.auth.client import AuthServiceClient

            client = AuthServiceClient("tracking_uri")
            client.create_user("newuser", "newpassword")
            ep = client.create_experiment_permission("myexperiment", "newuser", "READ")

            print(f"experiment_id: {ep.experiment_id}")
            print(f"user_id: {ep.user_id}")
            print(f"permission: {ep.permission}")

        .. code-block:: text
            :caption: Output

            experiment_id: myexperiment
            user_id: 3
            permission: READ
        """
        resp = self._request(
            CREATE_EXPERIMENT_PERMISSION,
            "POST",
            json={"experiment_id": experiment_id, "username": username, "permission": permission},
        )
        return ExperimentPermission.from_json(resp["experiment_permission"])

    def get_experiment_permission(self, experiment_id: str, username: str):
        """
        Get an experiment permission for a user.

        Args:
            experiment_id: The id of the experiment.
            username: The username.

        Raises:
            mlflow.exceptions.RestException: if the user does not exist,
                or no permission exists for this experiment user pair.
                Note that the default permission will still be effective even if
                no permission exists.

        Returns:
            A single :py:class:`mlflow.server.auth.entities.ExperimentPermission` object.

        .. code-block:: bash
            :caption: Example

            export MLFLOW_TRACKING_USERNAME=admin
            export MLFLOW_TRACKING_PASSWORD=password

        .. code-block:: python

            from mlflow.server.auth.client import AuthServiceClient

            client = AuthServiceClient("tracking_uri")
            client.create_user("newuser", "newpassword")
            client.create_experiment_permission("myexperiment", "newuser", "READ")
            ep = client.get_experiment_permission("myexperiment", "newuser")
            print(f"experiment_id: {ep.experiment_id}")
            print(f"user_id: {ep.user_id}")
            print(f"permission: {ep.permission}")

        .. code-block:: text
            :caption: Output

            experiment_id: myexperiment
            user_id: 3
            permission: READ
        """
        resp = self._request(
            GET_EXPERIMENT_PERMISSION,
            "GET",
            params={"experiment_id": experiment_id, "username": username},
        )
        return ExperimentPermission.from_json(resp["experiment_permission"])

    def update_experiment_permission(self, experiment_id: str, username: str, permission: str):
        """
        Update an existing experiment permission for a user.

        Args:
            experiment_id: The id of the experiment.
            username: The username.
            permission: New permission to grant. Must be one of "READ", "USE", "EDIT", "MANAGE" and
                "NO_PERMISSIONS".

        Raises:
            mlflow.exceptions.RestException: if the user does not exist, or no permission exists for
                this experiment user pair, or if the permission is invalid

        .. code-block:: bash
            :caption: Example

            export MLFLOW_TRACKING_USERNAME=admin
            export MLFLOW_TRACKING_PASSWORD=password

        .. code-block:: python

            from mlflow.server.auth.client import AuthServiceClient

            client = AuthServiceClient("tracking_uri")
            client.create_user("newuser", "newpassword")
            client.create_experiment_permission("myexperiment", "newuser", "READ")
            client.update_experiment_permission("myexperiment", "newuser", "EDIT")
        """
        self._request(
            UPDATE_EXPERIMENT_PERMISSION,
            "PATCH",
            json={"experiment_id": experiment_id, "username": username, "permission": permission},
        )

    def delete_experiment_permission(self, experiment_id: str, username: str):
        """
        Delete an existing experiment permission for a user.

        Args:
            experiment_id: The id of the experiment.
            username: The username.

        Raises:
            mlflow.exceptions.RestException: if the user does not exist, or no permission exists for
                this experiment user pair, or if the permission is invalid.
                Note that the default permission will still be effective even
                after the permission has been deleted.


        .. code-block:: bash
            :caption: Example

            export MLFLOW_TRACKING_USERNAME=admin
            export MLFLOW_TRACKING_PASSWORD=password

        .. code-block:: python

            from mlflow.server.auth.client import AuthServiceClient

            client = AuthServiceClient("tracking_uri")
            client.create_user("newuser", "newpassword")
            client.create_experiment_permission("myexperiment", "newuser", "READ")
            client.delete_experiment_permission("myexperiment", "newuser")
        """
        self._request(
            DELETE_EXPERIMENT_PERMISSION,
            "DELETE",
            json={"experiment_id": experiment_id, "username": username},
        )

    def create_registered_model_permission(self, name: str, username: str, permission: str):
        """
        Create a permission on an registered model for a user.

        Args:
            name: The name of the registered model.
            username: The username.
            permission: Permission to grant. Must be one of "READ", "USE", "EDIT", "MANAGE" and
                "NO_PERMISSIONS".

        Raises:
            mlflow.exceptions.RestException: if the user does not exist,
                or a permission already exists for this registered model user pair,
                or if the permission is invalid.
                Does not require ``name`` to be an existing registered model.

        Returns:
            A single :py:class:`mlflow.server.auth.entities.RegisteredModelPermission` object.
        """
        resp = self._request(
            CREATE_REGISTERED_MODEL_PERMISSION,
            "POST",
            json={"name": name, "username": username, "permission": permission},
        )
        return RegisteredModelPermission.from_json(resp["registered_model_permission"])

    def get_registered_model_permission(self, name: str, username: str):
        """
        Get an registered model permission for a user.

        Args:
            name: The name of the registered model.
            username: The username.

        Raises:
            mlflow.exceptions.RestException: if the user does not
                exist, or no permission exists for this registered model user pair. Note that the
                default permission will still be effective even if no permission exists.

        Returns:
             A single :py:class:`mlflow.server.auth.entities.RegisteredModelPermission` object.

        .. code-block:: bash
            :caption: Example

            export MLFLOW_TRACKING_USERNAME=admin
            export MLFLOW_TRACKING_PASSWORD=password

        .. code-block:: python

            from mlflow.server.auth.client import AuthServiceClient

            client = AuthServiceClient("tracking_uri")
            client.create_user("newuser", "newpassword")
            client.create_registered_model_permission("myregisteredmodel", "newuser", "READ")
            rmp = client.get_registered_model_permission("myregisteredmodel", "newuser")

            print(f"name: {rmp.name}")
            print(f"user_id: {rmp.user_id}")
            print(f"permission: {rmp.permission}")

        .. code-block:: text
            :caption: Output

            name: myregisteredmodel
            user_id: 3
            permission: READ
        """
        resp = self._request(
            GET_REGISTERED_MODEL_PERMISSION,
            "GET",
            params={"name": name, "username": username},
        )
        return RegisteredModelPermission.from_json(resp["registered_model_permission"])

    def update_registered_model_permission(self, name: str, username: str, permission: str):
        """
        Update an existing registered model permission for a user.

        Args:
            name: The name of the registered model.
            username: The username.
            permission: New permission to grant. Must be one of "READ", "USE", "EDIT", "MANAGE" and
                "NO_PERMISSIONS".

        Raises:
            mlflow.exceptions.RestException: if the user does not exist, or no permission exists for
                this registered model user pair, or if the permission is invalid.

        .. code-block:: bash
            :caption: Example

            export MLFLOW_TRACKING_USERNAME=admin
            export MLFLOW_TRACKING_PASSWORD=password

        .. code-block:: python

            from mlflow.server.auth.client import AuthServiceClient

            client = AuthServiceClient("tracking_uri")
            client.create_user("newuser", "newpassword")
            client.create_registered_model_permission("myregisteredmodel", "newuser", "READ")
            client.update_registered_model_permission("myregisteredmodel", "newuser", "EDIT")
        """
        self._request(
            UPDATE_REGISTERED_MODEL_PERMISSION,
            "PATCH",
            json={"name": name, "username": username, "permission": permission},
        )

    def delete_registered_model_permission(self, name: str, username: str):
        """
        Delete an existing registered model permission for a user.

        Args:
            name: The name of the registered model.
            username: The username.

        Raises:
            mlflow.exceptions.RestException: if the user does not exist,
                or no permission exists for this registered model user pair,
                or if the permission is invalid.
                Note that the default permission will still be effective even
                after the permission has been deleted.

        .. code-block:: bash
            :caption: Example

            export MLFLOW_TRACKING_USERNAME=admin
            export MLFLOW_TRACKING_PASSWORD=password

        .. code-block:: python

            from mlflow.server.auth.client import AuthServiceClient

            client = AuthServiceClient("tracking_uri")
            client.create_user("newuser", "newpassword")
            client.create_registered_model_permission("myregisteredmodel", "newuser", "READ")
            client.delete_registered_model_permission("myregisteredmodel", "newuser")
        """
        self._request(
            DELETE_REGISTERED_MODEL_PERMISSION,
            "DELETE",
            json={"name": name, "username": username},
        )

    def create_scorer_permission(
        self, experiment_id: str, scorer_name: str, username: str, permission: str
    ):
        """
        Create a permission on a scorer for a user.

        Args:
            experiment_id: The id of the experiment containing the scorer.
            scorer_name: The name of the scorer.
            username: The username.
            permission: Permission to grant. Must be one of "READ", "USE", "EDIT", "MANAGE" and
                "NO_PERMISSIONS".

        Raises:
            mlflow.exceptions.RestException: if the user does not exist,
                or the scorer permission already exists.

        Returns:
            A single :py:class:`mlflow.server.auth.entities.ScorerPermission` object.
        """
        resp = self._request(
            CREATE_SCORER_PERMISSION,
            "POST",
            json={
                "experiment_id": experiment_id,
                "scorer_name": scorer_name,
                "username": username,
                "permission": permission,
            },
        )
        return ScorerPermission.from_json(resp["scorer_permission"])

    def get_scorer_permission(self, experiment_id: str, scorer_name: str, username: str):
        """
        Get a scorer permission for a user.

        Args:
            experiment_id: The id of the experiment containing the scorer.
            scorer_name: The name of the scorer.
            username: The username.

        Raises:
            mlflow.exceptions.RestException: if the user does not exist,
                or no permission exists for this scorer user pair.

        Returns:
            A single :py:class:`mlflow.server.auth.entities.ScorerPermission` object.
        """
        resp = self._request(
            GET_SCORER_PERMISSION,
            "GET",
            params={
                "experiment_id": experiment_id,
                "scorer_name": scorer_name,
                "username": username,
            },
        )
        return ScorerPermission.from_json(resp["scorer_permission"])

    def update_scorer_permission(
        self, experiment_id: str, scorer_name: str, username: str, permission: str
    ):
        """
        Update an existing scorer permission for a user.

        Args:
            experiment_id: The id of the experiment containing the scorer.
            scorer_name: The name of the scorer.
            username: The username.
            permission: New permission to grant. Must be one of "READ", "USE", "EDIT", "MANAGE" and
                "NO_PERMISSIONS".

        Raises:
            mlflow.exceptions.RestException: if the user does not exist,
                or no permission exists for this scorer user pair,
                or if the permission is invalid.
        """
        self._request(
            UPDATE_SCORER_PERMISSION,
            "PATCH",
            json={
                "experiment_id": experiment_id,
                "scorer_name": scorer_name,
                "username": username,
                "permission": permission,
            },
        )

    def delete_scorer_permission(self, experiment_id: str, scorer_name: str, username: str):
        """
        Delete an existing scorer permission for a user.

        Args:
            experiment_id: The id of the experiment containing the scorer.
            scorer_name: The name of the scorer.
            username: The username.

        Raises:
            mlflow.exceptions.RestException: if the user does not exist,
                or no permission exists for this scorer user pair.
        """
        self._request(
            DELETE_SCORER_PERMISSION,
            "DELETE",
            json={
                "experiment_id": experiment_id,
                "scorer_name": scorer_name,
                "username": username,
            },
        )

    # Gateway secret permission methods

    def create_gateway_secret_permission(self, secret_id: str, username: str, permission: str):
        """
        Create a permission on a gateway secret for a user.

        Args:
            secret_id: The id of the gateway secret.
            username: The username.
            permission: Permission to grant. Must be one of "READ", "USE", "EDIT", "MANAGE" and
                "NO_PERMISSIONS".

        Returns:
            A single :py:class:`mlflow.server.auth.entities.GatewaySecretPermission` object.
        """
        resp = self._request(
            CREATE_GATEWAY_SECRET_PERMISSION,
            "POST",
            json={"secret_id": secret_id, "username": username, "permission": permission},
        )
        return GatewaySecretPermission.from_json(resp["gateway_secret_permission"])

    def get_gateway_secret_permission(self, secret_id: str, username: str):
        """
        Get a gateway secret permission for a user.

        Args:
            secret_id: The id of the gateway secret.
            username: The username.

        Returns:
            A single :py:class:`mlflow.server.auth.entities.GatewaySecretPermission` object.
        """
        resp = self._request(
            GET_GATEWAY_SECRET_PERMISSION,
            "GET",
            params={"secret_id": secret_id, "username": username},
        )
        return GatewaySecretPermission.from_json(resp["gateway_secret_permission"])

    def update_gateway_secret_permission(self, secret_id: str, username: str, permission: str):
        """
        Update an existing gateway secret permission for a user.

        Args:
            secret_id: The id of the gateway secret.
            username: The username.
            permission: New permission to grant. Must be one of "READ", "USE", "EDIT", "MANAGE" and
                "NO_PERMISSIONS".
        """
        self._request(
            UPDATE_GATEWAY_SECRET_PERMISSION,
            "PATCH",
            json={"secret_id": secret_id, "username": username, "permission": permission},
        )

    def delete_gateway_secret_permission(self, secret_id: str, username: str):
        """
        Delete an existing gateway secret permission for a user.

        Args:
            secret_id: The id of the gateway secret.
            username: The username.
        """
        self._request(
            DELETE_GATEWAY_SECRET_PERMISSION,
            "DELETE",
            json={"secret_id": secret_id, "username": username},
        )

    # Gateway endpoint permission methods

    def create_gateway_endpoint_permission(self, endpoint_id: str, username: str, permission: str):
        """
        Create a permission on a gateway endpoint for a user.

        Args:
            endpoint_id: The id of the gateway endpoint.
            username: The username.
            permission: Permission to grant. Must be one of "READ", "USE", "EDIT", "MANAGE" and
                "NO_PERMISSIONS".

        Returns:
            A single :py:class:`mlflow.server.auth.entities.GatewayEndpointPermission` object.
        """
        resp = self._request(
            CREATE_GATEWAY_ENDPOINT_PERMISSION,
            "POST",
            json={"endpoint_id": endpoint_id, "username": username, "permission": permission},
        )
        return GatewayEndpointPermission.from_json(resp["gateway_endpoint_permission"])

    def get_gateway_endpoint_permission(self, endpoint_id: str, username: str):
        """
        Get a gateway endpoint permission for a user.

        Args:
            endpoint_id: The id of the gateway endpoint.
            username: The username.

        Returns:
            A single :py:class:`mlflow.server.auth.entities.GatewayEndpointPermission` object.
        """
        resp = self._request(
            GET_GATEWAY_ENDPOINT_PERMISSION,
            "GET",
            params={"endpoint_id": endpoint_id, "username": username},
        )
        return GatewayEndpointPermission.from_json(resp["gateway_endpoint_permission"])

    def update_gateway_endpoint_permission(self, endpoint_id: str, username: str, permission: str):
        """
        Update an existing gateway endpoint permission for a user.

        Args:
            endpoint_id: The id of the gateway endpoint.
            username: The username.
            permission: New permission to grant. Must be one of "READ", "USE", "EDIT", "MANAGE" and
                "NO_PERMISSIONS".
        """
        self._request(
            UPDATE_GATEWAY_ENDPOINT_PERMISSION,
            "PATCH",
            json={"endpoint_id": endpoint_id, "username": username, "permission": permission},
        )

    def delete_gateway_endpoint_permission(self, endpoint_id: str, username: str):
        """
        Delete an existing gateway endpoint permission for a user.

        Args:
            endpoint_id: The id of the gateway endpoint.
            username: The username.
        """
        self._request(
            DELETE_GATEWAY_ENDPOINT_PERMISSION,
            "DELETE",
            json={"endpoint_id": endpoint_id, "username": username},
        )

    # Gateway model definition permission methods

    def create_gateway_model_definition_permission(
        self, model_definition_id: str, username: str, permission: str
    ):
        """
        Create a permission on a gateway model definition for a user.

        Args:
            model_definition_id: The id of the gateway model definition.
            username: The username.
            permission: Permission to grant. Must be one of "READ", "USE", "EDIT", "MANAGE" and
                "NO_PERMISSIONS".

        Returns:
            A single :py:class:`mlflow.server.auth.entities.GatewayModelDefinitionPermission`
            object.
        """
        resp = self._request(
            CREATE_GATEWAY_MODEL_DEFINITION_PERMISSION,
            "POST",
            json={
                "model_definition_id": model_definition_id,
                "username": username,
                "permission": permission,
            },
        )
        return GatewayModelDefinitionPermission.from_json(
            resp["gateway_model_definition_permission"]
        )

    def get_gateway_model_definition_permission(self, model_definition_id: str, username: str):
        """
        Get a gateway model definition permission for a user.

        Args:
            model_definition_id: The id of the gateway model definition.
            username: The username.

        Returns:
            A single :py:class:`mlflow.server.auth.entities.GatewayModelDefinitionPermission`
            object.
        """
        resp = self._request(
            GET_GATEWAY_MODEL_DEFINITION_PERMISSION,
            "GET",
            params={"model_definition_id": model_definition_id, "username": username},
        )
        return GatewayModelDefinitionPermission.from_json(
            resp["gateway_model_definition_permission"]
        )

    def update_gateway_model_definition_permission(
        self, model_definition_id: str, username: str, permission: str
    ):
        """
        Update an existing gateway model definition permission for a user.

        Args:
            model_definition_id: The id of the gateway model definition.
            username: The username.
            permission: New permission to grant. Must be one of "READ", "USE", "EDIT", "MANAGE" and
                "NO_PERMISSIONS".
        """
        self._request(
            UPDATE_GATEWAY_MODEL_DEFINITION_PERMISSION,
            "PATCH",
            json={
                "model_definition_id": model_definition_id,
                "username": username,
                "permission": permission,
            },
        )

    def delete_gateway_model_definition_permission(self, model_definition_id: str, username: str):
        """
        Delete an existing gateway model definition permission for a user.

        Args:
            model_definition_id: The id of the gateway model definition.
            username: The username.
        """
        self._request(
            DELETE_GATEWAY_MODEL_DEFINITION_PERMISSION,
            "DELETE",
            json={"model_definition_id": model_definition_id, "username": username},
        )

    def list_workspace_permissions(self, workspace_name: str) -> list[WorkspacePermission]:
        """
        List the permissions configured for the specified workspace.

        Args:
            workspace_name: The workspace name.

        Returns:
            A list of :py:class:`mlflow.server.auth.entities.WorkspacePermission` objects.
        """

        endpoint = self._workspace_endpoint(LIST_WORKSPACE_PERMISSIONS, workspace_name)
        resp = self._request(endpoint, "GET")
        return [WorkspacePermission.from_json(p) for p in resp["permissions"]]

    def set_workspace_permission(
        self, workspace_name: str, username: str, permission: str
    ) -> WorkspacePermission:
        """
        Create or update a workspace-level permission for a user.

        Args:
            workspace_name: The workspace name.
            username: The username receiving the permission.
            permission: Permission to grant. Must be one of "READ", "USE", "EDIT",
                "MANAGE", "NO_PERMISSIONS".

        Returns:
            A :py:class:`mlflow.server.auth.entities.WorkspacePermission` object.
        """

        endpoint = self._workspace_endpoint(LIST_WORKSPACE_PERMISSIONS, workspace_name)
        resp = self._request(
            endpoint,
            "POST",
            json={
                "username": username,
                "permission": permission,
            },
        )

        return WorkspacePermission.from_json(resp["permission"])

    def delete_workspace_permission(self, workspace_name: str, username: str) -> None:
        """
        Delete a workspace-level permission for a user.

        Args:
            workspace_name: The workspace name.
            username: The username whose permission should be removed.
        """

        endpoint = self._workspace_endpoint(LIST_WORKSPACE_PERMISSIONS, workspace_name)
        self._request(
            endpoint,
            "DELETE",
            params={"username": username},
            expected_status=204,
        )

    def list_user_workspace_permissions(self, username: str) -> list[WorkspacePermission]:
        """
        List workspace-level permissions assigned to a user.

        Args:
            username: The username.

        Returns:
            A list of :py:class:`mlflow.server.auth.entities.WorkspacePermission` objects.
        """

        resp = self._request(
            LIST_USER_WORKSPACE_PERMISSIONS,
            "GET",
            params={"username": username},
        )
        return [WorkspacePermission.from_json(p) for p in resp["permissions"]]

    # ---- Role management (RBAC) ----

    def create_role(
        self,
        workspace: str,
        name: str,
        description: str | None = None,
    ) -> Role:
        payload = {"workspace": workspace, "name": name}
        if description is not None:
            payload["description"] = description
        resp = self._request(CREATE_ROLE, "POST", json=payload)
        return Role.from_json(resp["role"])

    def get_role(self, role_id: int) -> Role:
        resp = self._request(GET_ROLE, "GET", params={"role_id": str(role_id)})
        return Role.from_json(resp["role"])

    def list_roles(self, workspace: str) -> list[Role]:
        resp = self._request(LIST_ROLES, "GET", params={"workspace": workspace})
        return [Role.from_json(r) for r in resp["roles"]]

    def update_role(
        self, role_id: int, name: str | None = None, description: str | None = None
    ) -> Role:
        payload: dict[str, object] = {"role_id": role_id}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        resp = self._request(UPDATE_ROLE, "PATCH", json=payload)
        return Role.from_json(resp["role"])

    def delete_role(self, role_id: int) -> None:
        self._request(DELETE_ROLE, "DELETE", json={"role_id": role_id})

    def add_role_permission(
        self, role_id: int, resource_type: str, resource_pattern: str, permission: str
    ) -> RolePermission:
        resp = self._request(
            ADD_ROLE_PERMISSION,
            "POST",
            json={
                "role_id": role_id,
                "resource_type": resource_type,
                "resource_pattern": resource_pattern,
                "permission": permission,
            },
        )
        return RolePermission.from_json(resp["role_permission"])

    def remove_role_permission(self, role_permission_id: int) -> None:
        self._request(
            REMOVE_ROLE_PERMISSION, "DELETE", json={"role_permission_id": role_permission_id}
        )

    def list_role_permissions(self, role_id: int) -> list[RolePermission]:
        resp = self._request(LIST_ROLE_PERMISSIONS, "GET", params={"role_id": str(role_id)})
        return [RolePermission.from_json(p) for p in resp["role_permissions"]]

    def update_role_permission(self, role_permission_id: int, permission: str) -> RolePermission:
        resp = self._request(
            UPDATE_ROLE_PERMISSION,
            "PATCH",
            json={"role_permission_id": role_permission_id, "permission": permission},
        )
        return RolePermission.from_json(resp["role_permission"])

    def assign_role(self, username: str, role_id: int) -> UserRoleAssignment:
        resp = self._request(ASSIGN_ROLE, "POST", json={"username": username, "role_id": role_id})
        return UserRoleAssignment.from_json(resp["assignment"])

    def unassign_role(self, username: str, role_id: int) -> None:
        self._request(UNASSIGN_ROLE, "DELETE", json={"username": username, "role_id": role_id})

    def list_user_roles(self, username: str) -> list[Role]:
        resp = self._request(LIST_USER_ROLES, "GET", params={"username": username})
        return [Role.from_json(r) for r in resp["roles"]]

    def list_role_users(self, role_id: int) -> list[UserRoleAssignment]:
        resp = self._request(LIST_ROLE_USERS, "GET", params={"role_id": str(role_id)})
        return [UserRoleAssignment.from_json(a) for a in resp["assignments"]]

    def list_all_roles(self) -> list[Role]:
        # Same endpoint as list_roles; omitting the ``workspace`` param returns the
        # cross-workspace listing (admin-only, enforced server-side).
        resp = self._request(LIST_ROLES, "GET")
        return [Role.from_json(r) for r in resp["roles"]]
