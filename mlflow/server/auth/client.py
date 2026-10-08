from mlflow.server.auth.entities import (
    GetUserPermissionResult,
    MutationConditions,
    Role,
    RolePermission,
    User,
    UserRoleAssignment,
)
from mlflow.server.auth.routes import (
    ADD_MUTATION_CONDITIONS,
    ADD_ROLE_PERMISSION,
    ADD_USER_MUTATION_CONDITION,
    ASSIGN_ROLE,
    CREATE_ROLE,
    CREATE_USER,
    DELETE_ROLE,
    DELETE_USER,
    GET_MUTATION_CONDITIONS,
    GET_ROLE,
    GET_USER,
    GET_USER_PERMISSION,
    GRANT_USER_PERMISSION,
    LIST_MUTATION_CONDITIONS,
    LIST_ROLE_PERMISSIONS,
    LIST_ROLE_USERS,
    LIST_ROLES,
    LIST_USER_ROLES,
    REMOVE_MUTATION_CONDITIONS,
    REMOVE_ROLE_PERMISSION,
    REVOKE_USER_PERMISSION,
    UNASSIGN_ROLE,
    UPDATE_MUTATION_CONDITIONS,
    UPDATE_ROLE,
    UPDATE_ROLE_PERMISSION,
    UPDATE_USER_ADMIN,
    UPDATE_USER_PASSWORD,
)
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.rest_utils import http_request, verify_rest_response

#: Distinguishes "leave this condition unchanged" from "clear it". ``None`` already
#: means clear, so omission needs its own marker -- otherwise an update touching one
#: condition would silently drop the other.
_UNSET = object()


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
        """Grant ``permission`` on resources of ``resource_type`` matching ``resource_pattern``.

        Args:
            role_id: The role to add the permission to.
            resource_type: One of the container types -- ``experiment``, ``registered_model``,
                ``prompt``, ``scorer``, ``gateway_secret``, ``gateway_endpoint``,
                ``gateway_model_definition``, ``mcp_server`` -- or ``workspace``, or one of the
                sub-resource tiers: ``run``, ``trace``, ``assessment``, ``logged_model``,
                ``review_queue``, ``registered_model_version``, ``prompt_version``,
                ``scorer_version``, ``mcp_server_version``.
            resource_pattern: A resource id or name, or ``"*"`` for every resource of the type.
                ``workspace`` and the sub-resource tiers accept only ``"*"``; a specific id on
                those is rejected, because a per-id grant would be honoured on a point route and
                silently lapse in a search.
            permission: ``READ``, ``USE``, ``EDIT``, ``MANAGE``, or ``DENY``. ``DENY`` is an
                explicit-deny override that wins over any positive grant it is folded with, and
                applies at the tier it is granted on. ``workspace`` accepts only ``USE`` and
                ``MANAGE``.

        Returns:
            The created :py:class:`RolePermission`.
        """
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

    # ---- Mutation conditions (condition-based access control) ----
    #
    # Two optional filters per (role, resource_type) that gate create/mutation only,
    # never reads. Conditions **subtract** from what the role's grants allow: they
    # never confer access, and a role with no conditions behaves exactly as before.

    def add_mutation_condition(
        self,
        role_id: int,
        resource_type: str,
        *,
        resource_pattern: str | None = None,
        container_resource_type: str | None = None,
        container_resource_pattern: str | None = None,
        value_condition: str | None = None,
        target_condition: str | None = None,
    ) -> MutationConditions:
        """Attach one condition object to ``role_id`` for ``resource_type``.

        A role may hold several conditions per resource type. **Every applicable one
        must pass** -- they AND, they are not alternatives. Two objects saying
        "``env = dev``" and "``owner = me``" permit only resources that are both, which
        is the most common way to lock yourself out.

        Args:
            role_id: The role to condition. Conditions apply on top of that role's
                grants and cannot widen them.
            resource_type: One of ``experiment``, ``run``, ``trace``, ``logged_model``,
                ``registered_model``, ``registered_model_version``, ``prompt``,
                ``prompt_version``, ``mcp_server``, ``mcp_server_version``. Other types
                are rejected -- a condition is only meaningful for a type carrying tags
                or aliases.
            resource_pattern: Which resources of ``resource_type`` to govern: ``"*"``
                for all (the default), or one resource id. The grain follows that type's
                *grants*, so a top-level type (``experiment``, ``registered_model``,
                ``prompt``, ``mcp_server``) accepts an id while a sub-resource is
                wildcard-only -- a per-id child restriction could not be enforced in list
                and search paths, so it is refused rather than half-held.
            container_resource_type: Which container to govern within. ``"workspace"``
                (the default) is no narrowing; otherwise the type's declared parent. A
                top-level type has no other container, which is why it narrows with
                ``resource_pattern`` instead. Containment is exact and does not inherit
                across resource types.
            container_resource_pattern: The container's id, or ``"*"``. A wildcard here
                means every container, which is the same statement as ``"workspace"`` and
                is normalised to it.
            value_condition: Constrains *what values* a mutation may set, as a filter
                string over ``tag_key``, ``tag_value`` and ``alias``. Evaluated against
                the request, and applied on create. A clause whose identifier the
                request does not carry is vacuous, so a metrics-only ``LogBatch`` is
                not denied by a ``tag_key`` clause. Reserved ``mlflow.*`` tag keys are
                rejected here, because MLflow writes them itself.
            target_condition: Constrains *which existing resources* may be mutated, as
                a filter string over ``tags.<key>`` and ``aliases.<name>``. Evaluated
                against the resource's current state, and vacuous on create. **A
                missing tag fails the clause**, matching search semantics -- so
                ``tags.lifecycle != 'prod'`` denies an untagged resource.

        At least one of the two filters is required: an object with neither restricts
        nothing while reading as a configured restriction. Each takes at most five
        AND-joined clauses; ``OR`` is not supported. Use ``!=`` or ``NOT IN`` to
        exclude.

        Returns:
            The created :py:class:`MutationConditions`, carrying the server-allocated
            ``condition_slot``.
        """
        resp = self._request(
            ADD_MUTATION_CONDITIONS,
            "POST",
            json={
                "role_id": role_id,
                "resource_type": resource_type,
                "resource_pattern": resource_pattern,
                "container_resource_type": container_resource_type,
                "container_resource_pattern": container_resource_pattern,
                "value_condition": value_condition,
                "target_condition": target_condition,
            },
        )
        return MutationConditions.from_json(resp["mutation_conditions"])

    def add_user_mutation_condition(
        self,
        username: str,
        resource_type: str,
        *,
        resource_pattern: str | None = None,
        container_resource_type: str | None = None,
        container_resource_pattern: str | None = None,
        value_condition: str | None = None,
        target_condition: str | None = None,
    ) -> MutationConditions:
        """Attach one condition object to ``username``'s direct grants.

        The user-addressed counterpart of :meth:`add_mutation_condition`, and the
        condition analogue of :meth:`grant_user_permission`. Per-user access is stored on
        a hidden per-user role; this resolves that role -- creating it if the user has no
        direct grants yet -- so the caller never names it.

        Everything :meth:`add_mutation_condition` says about semantics applies unchanged:
        conditions AND with each other and with those on the user's other roles, they
        cannot widen a grant, and they never apply to reads.

        Args:
            username: The user whose direct grants to condition.
            resource_type: As :meth:`add_mutation_condition`.
            resource_pattern: As :meth:`add_mutation_condition`.
            container_resource_type: As :meth:`add_mutation_condition`.
            container_resource_pattern: As :meth:`add_mutation_condition`.
            value_condition: As :meth:`add_mutation_condition`.
            target_condition: As :meth:`add_mutation_condition`.

        Returns:
            The created :py:class:`MutationConditions`. Its ``role_id`` is the per-user
            role, which is an implementation detail -- address later updates and removals
            by the returned ``id``.
        """
        resp = self._request(
            ADD_USER_MUTATION_CONDITION,
            "POST",
            json={
                "username": username,
                "resource_type": resource_type,
                "resource_pattern": resource_pattern,
                "container_resource_type": container_resource_type,
                "container_resource_pattern": container_resource_pattern,
                "value_condition": value_condition,
                "target_condition": target_condition,
            },
        )
        return MutationConditions.from_json(resp["mutation_conditions"])

    def get_mutation_condition(self, condition_id: int) -> MutationConditions:
        resp = self._request(
            GET_MUTATION_CONDITIONS,
            "GET",
            params={"condition_id": str(condition_id)},
        )
        return MutationConditions.from_json(resp["mutation_conditions"])

    def update_mutation_condition(
        self,
        condition_id: int,
        *,
        value_condition: str | None = _UNSET,
        target_condition: str | None = _UNSET,
        resource_pattern: str | None = _UNSET,
        container_resource_type: str | None = _UNSET,
        container_resource_pattern: str | None = _UNSET,
    ) -> MutationConditions | None:
        """Update one condition object, leaving any argument you omit untouched.

        Passing ``None`` explicitly **clears** that condition; omitting the argument
        leaves it as it is. The two are different operations, so they cannot share a
        sentinel -- if they did, an update that only meant to change the value
        condition would silently drop the target condition.

        The scope moves as a unit: pass any of ``resource_pattern``,
        ``container_resource_type`` or ``container_resource_pattern`` and all three are
        replaced, with anything you omit taking its widest default. The axes are validated
        together -- a container's legality depends on the resource type -- so they cannot be
        changed independently.

        Returns:
            The updated object, or ``None`` if clearing both filters deleted it -- an
            object restricting nothing is removed rather than left holding a slot.
        """
        body: dict[str, object] = {"condition_id": condition_id}
        if value_condition is not _UNSET:
            body["value_condition"] = value_condition
        if target_condition is not _UNSET:
            body["target_condition"] = target_condition
        scope_args = (resource_pattern, container_resource_type, container_resource_pattern)
        if any(arg is not _UNSET for arg in scope_args):
            # The scope moves as a unit -- the server validates the axes together -- so an
            # argument left out here takes its default rather than its current value.
            body["update_scope"] = True
            body["resource_pattern"] = None if resource_pattern is _UNSET else resource_pattern
            body["container_resource_type"] = (
                None if container_resource_type is _UNSET else container_resource_type
            )
            body["container_resource_pattern"] = (
                None if container_resource_pattern is _UNSET else container_resource_pattern
            )
        resp = self._request(UPDATE_MUTATION_CONDITIONS, "PATCH", json=body)
        payload = resp.get("mutation_conditions")
        return MutationConditions.from_json(payload) if payload else None

    def remove_mutation_condition(self, condition_id: int) -> None:
        self._request(
            REMOVE_MUTATION_CONDITIONS,
            "DELETE",
            json={"condition_id": condition_id},
        )

    def list_mutation_conditions(self, role_id: int) -> list[MutationConditions]:
        resp = self._request(LIST_MUTATION_CONDITIONS, "GET", params={"role_id": str(role_id)})
        return [MutationConditions.from_json(c) for c in resp["mutation_conditions"]]

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
