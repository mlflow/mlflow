from mlflow.server.auth.entities import (
    ExperimentPermission,
    RegisteredModelPermission,
    ScorerPermission,
    User,
)
from mlflow.server.auth.routes import (
    CREATE_EXPERIMENT_PERMISSION,
    CREATE_REGISTERED_MODEL_PERMISSION,
    CREATE_SCORER_PERMISSION,
    CREATE_USER,
    DELETE_EXPERIMENT_PERMISSION,
    DELETE_REGISTERED_MODEL_PERMISSION,
    DELETE_SCORER_PERMISSION,
    DELETE_USER,
    GET_EXPERIMENT_PERMISSION,
    GET_REGISTERED_MODEL_PERMISSION,
    GET_SCORER_PERMISSION,
    GET_USER,
    UPDATE_EXPERIMENT_PERMISSION,
    UPDATE_REGISTERED_MODEL_PERMISSION,
    UPDATE_SCORER_PERMISSION,
    UPDATE_USER_ADMIN,
    UPDATE_USER_PASSWORD,
)
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.rest_utils import http_request_safe


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

    def _request(self, endpoint, method, **kwargs):
        host_creds = get_default_host_creds(self.tracking_uri)
        resp = http_request_safe(host_creds, endpoint, method, **kwargs)
        return resp.json()

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

    def update_user_password(self, username: str, password: str):
        """
        Update the password of a specific user.

        Args:
            username: The username.
            password: The new password.

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

            client.update_user_password("newuser", "anotherpassword")
        """
        self._request(
            UPDATE_USER_PASSWORD,
            "PATCH",
            json={"username": username, "password": password},
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
            permission: Permission to grant. Must be one of "READ", "EDIT", "MANAGE" and
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
            permission: New permission to grant. Must be one of "READ", "EDIT", "MANAGE" and
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
            permission: Permission to grant. Must be one of "READ", "EDIT", "MANAGE" and
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
            permission: New permission to grant. Must be one of "READ", "EDIT", "MANAGE" and
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
            permission: Permission to grant. Must be one of "READ", "EDIT", "MANAGE" and
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
            permission: New permission to grant. Must be one of "READ", "EDIT", "MANAGE" and
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
