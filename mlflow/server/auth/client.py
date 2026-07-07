from mlflow.server.auth.entities import (
    GetUserPermissionResult,
    Role,
    RolePermission,
    User,
    UserRoleAssignment,
)
from mlflow.server.auth.routes import (
    ADD_ROLE_PERMISSION,
    ASSIGN_ROLE,
    CREATE_ROLE,
    CREATE_USER,
    DELETE_ROLE,
    DELETE_USER,
    GET_ROLE,
    GET_USER,
    GET_USER_PERMISSION,
    GRANT_USER_PERMISSION,
    LIST_ROLE_PERMISSIONS,
    LIST_ROLE_USERS,
    LIST_ROLES,
    LIST_USER_ROLES,
    REMOVE_ROLE_PERMISSION,
    REVOKE_USER_PERMISSION,
    UNASSIGN_ROLE,
    UPDATE_ROLE,
    UPDATE_ROLE_PERMISSION,
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

    # ---- Unified per-user permission convenience APIs ----
    # Grant / revoke / check one resource permission for a user. Preserve the
    # legacy per-resource MANAGE delegation (per-resource MANAGE gates writes)
    # via a uniform ``(resource_type, resource_id)`` shape.

    def grant_user_permission(
        self,
        username: str,
        resource_type: str,
        resource_id: str,
        permission: str,
    ) -> None:
        self._request(
            GRANT_USER_PERMISSION,
            "POST",
            json={
                "username": username,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "permission": permission,
            },
        )

    def revoke_user_permission(self, username: str, resource_type: str, resource_id: str) -> None:
        self._request(
            REVOKE_USER_PERMISSION,
            "POST",
            json={
                "username": username,
                "resource_type": resource_type,
                "resource_id": resource_id,
            },
        )

    def get_user_permission(
        self,
        username: str,
        resource_type: str,
        resource_id: str,
    ) -> GetUserPermissionResult:
        resp = self._request(
            GET_USER_PERMISSION,
            "GET",
            params={
                "username": username,
                "resource_type": resource_type,
                "resource_id": resource_id,
            },
        )
        return GetUserPermissionResult.from_json(resp)
