import logging
import re
from collections.abc import Collection, Iterable, Mapping
from typing import NamedTuple
from urllib.parse import quote, unquote

from sqlalchemy import and_, or_, select, text
from sqlalchemy.exc import IntegrityError, MultipleResultsFound, NoResultFound
from sqlalchemy.orm import selectinload, sessionmaker
from werkzeug.security import check_password_hash, generate_password_hash

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    TEMPORARILY_UNAVAILABLE,
)
from mlflow.server.auth.conditions import (
    CONTAINER_WORKSPACE,
    MAX_CONDITIONS_PER_ROLE_TYPE,
    NAMESPACE_REQUEST,
    NAMESPACE_RESOURCE,
    PARENT_RESOURCE_TYPES,
    normalize_condition_scope,
    validate_condition,
    validate_condition_resource_type,
)
from mlflow.server.auth.db import utils as dbutils
from mlflow.server.auth.db.models import (
    SqlMutationConditions,
    SqlRole,
    SqlRolePermission,
    SqlUser,
    SqlUserRoleAssignment,
)
from mlflow.server.auth.entities import (
    ExperimentPermission,
    GatewayEndpointPermission,
    GatewayModelDefinitionPermission,
    GatewaySecretPermission,
    MCPServerPermission,
    MutationConditions,
    RegisteredModelPermission,
    Role,
    RolePermission,
    ScorerPermission,
    User,
    UserRoleAssignment,
    WorkspacePermission,
)
from mlflow.server.auth.permissions import (
    DENY,
    MANAGE,
    RESOURCE_TYPE_EXPERIMENT,
    RESOURCE_TYPE_GATEWAY_ENDPOINT,
    RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION,
    RESOURCE_TYPE_GATEWAY_SECRET,
    RESOURCE_TYPE_MCP_SERVER,
    RESOURCE_TYPE_PROMPT,
    RESOURCE_TYPE_PROMPT_VERSION,
    RESOURCE_TYPE_REGISTERED_MODEL,
    RESOURCE_TYPE_REGISTERED_MODEL_VERSION,
    RESOURCE_TYPE_SCORER,
    RESOURCE_TYPE_WORKSPACE,
    Permission,
    _validate_permission_for_resource_type,
    _validate_resource_pattern,
    _validate_resource_type,
    get_permission,
    max_permission,
)
from mlflow.store.db.utils import (
    _get_managed_session_maker,
    _get_routing_session_maker,
    create_sqlalchemy_engine_with_retry,
)
from mlflow.utils import workspace_context
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import _validate_password, _validate_username
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

_logger = logging.getLogger(__name__)

# Pre-RBAC permission tables retained on disk by
# ``e5f6a7b8c9d0_migrate_permissions_to_roles``. The runtime no longer reads or
# writes these tables — they exist solely so operators can roll back the
# simplification migration without restoring from backup. Their FKs to
# ``users.id`` are non-cascading from earlier migrations
# (``8606fa83a998_initial_migration``), so ``delete_user`` must scrub the user's
# rows here before deleting the user row itself.
#
# GRADUATION LEVER. When the drop migration
# (``f6a7b8c9d0e1_drop_legacy_permission_tables``, scheduled for MLflow 3.X+2 —
# tracking: https://github.com/mlflow/mlflow/issues/23087) lands and removes
# these tables from the schema, this tuple becomes ``()`` and the loop in
# ``delete_user`` below becomes a no-op. The graduation PR should:
#   1. Add ``f6a7b8c9d0e1_drop_legacy_permission_tables`` that ``op.drop_table``s
#      every entry in this tuple.
#   2. Set this tuple to ``()`` (or remove the constant + loop entirely).
#   3. Update ``test_auth_and_tracking_store_coexist`` and
#      ``test_upgrade_from_legacy_database`` to assert the tables are absent
#      post-upgrade.
#   4. Drop ``test_delete_user_clears_retained_legacy_permission_rows`` —
#      cleanup-by-cascade is sufficient once the legacy FKs are gone.
_RETAINED_LEGACY_PERMISSION_TABLES: tuple[str, ...] = (
    "experiment_permissions",
    "registered_model_permissions",
    "scorer_permissions",
    "gateway_secret_permissions",
    "gateway_endpoint_permissions",
    "gateway_model_definition_permissions",
    "workspace_permissions",
)


class RoleGrantRow(NamedTuple):
    """One role-based grant row, detached from the ORM session.

    A plain tuple deliberately: the fold runs outside the store, so nothing it receives
    should be a live SQLAlchemy instance.
    """

    resource_type: str
    resource_pattern: str
    permission: str


class MutationConditionRow(NamedTuple):
    """One role's conditions for one resource type, detached from the ORM session.

    A plain tuple for the same reason as ``RoleGrantRow``: evaluation runs outside
    the store, so nothing it receives should be a live SQLAlchemy instance. Both
    fields are the filter string as authored -- parsing happens in
    ``mlflow.server.auth.conditions``, not here.
    """

    resource_type: str
    value_condition: str | None
    target_condition: str | None
    #: ``"*"`` for every resource of the type, or the one id this row governs. Matched by
    #: the gate rather than in SQL: a cascade's child ids are not known when this query
    #: runs, so filtering here would drop a condition written to govern one of those
    #: children -- a fail-open. The container axis *is* filtered in SQL, because the
    #: container is always resolved before the query runs.
    resource_pattern: str = "*"


#: How many times ``add_mutation_condition`` re-picks a slot before giving up.
#:
#: Each retry means a concurrent writer committed the slot this transaction chose, so
#: the loop only spins while adds are genuinely racing. A small bound is right: the
#: alternative to retrying is overshooting the per-role-and-type limit, and a caller
#: who loses this many races is better served by an explicit "retry the request" than
#: by an unbounded loop holding a write transaction open.
_SLOT_ALLOCATION_ATTEMPTS = 5


class SqlAlchemyStore:
    @classmethod
    def _get_active_workspace_name(cls) -> str:
        if not MLFLOW_ENABLE_WORKSPACES.get():
            return DEFAULT_WORKSPACE_NAME

        if workspace_name := workspace_context.get_request_workspace():
            return workspace_name

        raise MlflowException.invalid_parameter_value(
            "Active workspace is required. Configure a default workspace or call "
            "mlflow.set_workspace() before interacting with the authentication store."
        )

    def init_db(self, db_uri, read_db_uri=None):
        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.engine = create_sqlalchemy_engine_with_retry(db_uri)
        dbutils.migrate_if_needed(self.engine, "head")

        # Set up read replica engine if provided
        if read_db_uri and read_db_uri != db_uri:
            self.read_engine = create_sqlalchemy_engine_with_retry(read_db_uri)
            WriteSessionMaker = sessionmaker(bind=self.engine)
            ReadSessionMaker = sessionmaker(bind=self.read_engine)
            self.ManagedSessionMaker = _get_routing_session_maker(
                WriteSessionMaker, ReadSessionMaker, self.db_type
            )
        else:
            if read_db_uri and read_db_uri == db_uri:
                _logger.warning(
                    "read_db_uri is the same as the primary db_uri; "
                    "read replica routing will not be enabled. "
                    "This is likely a configuration mistake."
                )
            self.read_engine = None
            SessionMaker = sessionmaker(bind=self.engine)
            self.ManagedSessionMaker = _get_managed_session_maker(SessionMaker, self.db_type)

    def authenticate_user(self, username: str, password: str, *, use_primary: bool = False) -> bool:
        with self.ManagedSessionMaker(read_only=not use_primary) as session:
            try:
                user = self._get_user(session, username)
                return check_password_hash(user.password_hash, password)
            except MlflowException:
                return False

    def create_user(self, username: str, password: str, is_admin: bool = False) -> User:
        _validate_username(username)
        _validate_password(password)
        pwhash = generate_password_hash(password)
        with self.ManagedSessionMaker(read_only=False) as session:
            try:
                user = SqlUser(username=username, password_hash=pwhash, is_admin=is_admin)
                session.add(user)
                session.flush()
                return user.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"User (username={username}) already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                ) from e

    @staticmethod
    def _get_user(session, username: str) -> SqlUser:
        try:
            return session.query(SqlUser).filter(SqlUser.username == username).one()
        except NoResultFound:
            raise MlflowException(
                f"User with username={username} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                f"Found multiple users with username={username}",
                INVALID_STATE,
            )

    def has_user(self, username: str, *, use_primary: bool = False) -> bool:
        with self.ManagedSessionMaker(read_only=not use_primary) as session:
            return session.query(SqlUser).filter(SqlUser.username == username).first() is not None

    def get_user(self, username: str) -> User:
        with self.ManagedSessionMaker() as session:
            return self._get_user(session, username).to_mlflow_entity()

    def list_users(self) -> list[User]:
        with self.ManagedSessionMaker() as session:
            users = session.query(SqlUser).all()
            return [u.to_mlflow_entity() for u in users]

    def list_users_with_roles(self) -> list[tuple[User, list[Role]]]:
        """
        Return every user paired with their role assignments. Eager-loads
        assignments / roles / role-permissions in batched queries so the admin
        Users tab can render per-user role info without N per-row requests.
        """
        with self.ManagedSessionMaker() as session:
            users = (
                session
                .query(SqlUser)
                .options(
                    selectinload(SqlUser.user_role_assignments)
                    .selectinload(SqlUserRoleAssignment.role)
                    .selectinload(SqlRole.permissions)
                )
                .all()
            )
            return [
                (u.to_mlflow_entity(), [a.role.to_mlflow_entity() for a in u.user_role_assignments])
                for u in users
            ]

    def update_user(
        self, username: str, password: str | None = None, is_admin: bool | None = None
    ) -> User:
        with self.ManagedSessionMaker(read_only=False) as session:
            user = self._get_user(session, username)
            if password is not None:
                _validate_password(password)
                pwhash = generate_password_hash(password)
                user.password_hash = pwhash
            if is_admin is not None:
                user.is_admin = is_admin
            return user.to_mlflow_entity()

    def delete_user(self, username: str):
        with self.ManagedSessionMaker(read_only=False) as session:
            user = self._get_user(session, username)
            # The user's per-resource grants live as role_permissions rows under a
            # synthetic `__user_<id>__` role. Delete those first so their assignments
            # and permissions don't block the user-row delete on strict FK backends.
            synthetic_name = self._synthetic_user_role_name(user.id)
            for role in session.query(SqlRole).filter(SqlRole.name == synthetic_name).all():
                session.delete(role)
            # Scrub the user's rows from the retained legacy permission tables.
            # See ``_RETAINED_LEGACY_PERMISSION_TABLES`` (top of module) for why
            # this loop exists and the steps to retire it once the drop migration
            # ships. When that constant is empty, this is a no-op.
            for table in _RETAINED_LEGACY_PERMISSION_TABLES:
                session.execute(
                    text(f"DELETE FROM {table} WHERE user_id = :uid"),
                    {"uid": user.id},
                )
            session.flush()
            session.delete(user)

    # ---- Synthetic user-role helpers ----
    #
    # The auth system's user-facing permission shape (e.g. `get_experiment_permission`) is
    # expressed as "user X has permission P on resource R". The underlying storage is
    # ``role_permissions``, which binds grants to a role rather than to a user directly.
    # We bridge the two by maintaining a hidden per-user role named `__user_<user_id>__`
    # in each workspace the user has grants in, assigning the user to it, and attaching
    # their per-resource grants as rows on that role.
    #
    # The synthetic namespace (``__user_<int>__``) is RESERVED: ``create_role`` and
    # ``update_role`` reject names matching the pattern so an API consumer can't hijack a
    # synthetic role to receive another user's grants. Bulk cleanup / rename helpers
    # also restrict their scope to synthetic roles so they never touch admin-created
    # roles that happen to hold overlapping grants.
    _SYNTHETIC_ROLE_PREFIX = "__user_"
    _SYNTHETIC_ROLE_SUFFIX = "__"
    _SYNTHETIC_ROLE_NAME_RE = re.compile(r"^__user_\d+__$")

    @classmethod
    def _synthetic_user_role_name(cls, user_id: int) -> str:
        return f"{cls._SYNTHETIC_ROLE_PREFIX}{user_id}{cls._SYNTHETIC_ROLE_SUFFIX}"

    @classmethod
    def _is_synthetic_role_name(cls, name: str | None) -> bool:
        return name is not None and cls._SYNTHETIC_ROLE_NAME_RE.match(name) is not None

    @classmethod
    def _reject_synthetic_role_name(cls, name: str) -> None:
        """Guard user-facing role CRUD against the reserved synthetic namespace.

        Reject any name starting with ``__user_`` (not just the strict
        ``__user_<digits>__`` synthetic pattern): the whole prefix is reserved
        so a future change to the synthetic naming scheme can't be hijacked by
        an admin-created role that lined up with the older format.
        """
        if name.startswith(cls._SYNTHETIC_ROLE_PREFIX):
            raise MlflowException.invalid_parameter_value(
                f"Role name {name!r} uses the reserved '__user_' prefix, which "
                "is held for the per-user permission representation. Choose a "
                "different name."
            )

    def _get_or_create_synthetic_user_role(self, session, user_id: int, workspace: str) -> SqlRole:
        # Race-proof: two concurrent permission mutations for the same (user, workspace)
        # can both see "role doesn't exist yet" and both try to INSERT. Wrap the INSERT
        # in a SAVEPOINT so the UniqueConstraint violation rolls back only that nested
        # scope and we can recover by re-querying the winner's row. Same pattern for
        # the user->role assignment.
        name = self._synthetic_user_role_name(user_id)
        role = (
            session
            .query(SqlRole)
            .filter(SqlRole.workspace == workspace, SqlRole.name == name)
            .first()
        )
        if role is None:
            try:
                with session.begin_nested():
                    role = SqlRole(name=name, workspace=workspace, description=None)
                    session.add(role)
                    session.flush()
            except IntegrityError:
                # No other-assignee check needed here: the SAVEPOINT only
                # rolls back when our own INSERT lost the race for the same
                # ``(workspace, __user_<user_id>__)`` tuple, so the recovered
                # row is guaranteed to be this user's synthetic role.
                role = (
                    session
                    .query(SqlRole)
                    .filter(SqlRole.workspace == workspace, SqlRole.name == name)
                    .one()
                )
        else:
            # Defense-in-depth: if a role with this reserved synthetic name already
            # exists but has assignments for other users, we would leak grants
            # across accounts by attaching this user's grants to it. This shouldn't
            # happen in practice (``create_role``/``update_role`` reject the
            # synthetic pattern), but on databases that predate the reservation or
            # after a manual SQL insert it would slip through. Refuse to proceed.
            other_assignee = (
                session
                .query(SqlUserRoleAssignment.user_id)
                .filter(
                    SqlUserRoleAssignment.role_id == role.id,
                    SqlUserRoleAssignment.user_id != user_id,
                )
                .first()
            )
            if other_assignee is not None:
                raise MlflowException.invalid_parameter_value(
                    f"Role {name!r} in workspace {workspace!r} collides with the "
                    "reserved '__user_<id>__' synthetic namespace but is already "
                    f"assigned to user_id={other_assignee[0]}. Rename or delete "
                    "the conflicting role before granting per-user permissions."
                )
        assignment = (
            session
            .query(SqlUserRoleAssignment)
            .filter(
                SqlUserRoleAssignment.user_id == user_id,
                SqlUserRoleAssignment.role_id == role.id,
            )
            .first()
        )
        if assignment is None:
            try:
                with session.begin_nested():
                    session.add(SqlUserRoleAssignment(user_id=user_id, role_id=role.id))
                    session.flush()
            except IntegrityError:
                # Another concurrent write already assigned the user.
                pass
        return role

    def _synthetic_role_ids(self, session, workspace: str | None = None) -> list[int]:
        """
        Return ids of synthetic ``__user_<id>__`` roles, optionally scoped to
        ``workspace``. Bulk cleanup/rename helpers route through this so they never
        touch admin-created roles that happen to share a grant.
        """
        query = session.query(SqlRole.id, SqlRole.name)
        if workspace is not None:
            query = query.filter(SqlRole.workspace == workspace)
        return [rid for (rid, name) in query.all() if self._is_synthetic_role_name(name)]

    # ---- Per-user grant helpers ----
    #
    # Generic API for per-user permission grants on the user's synthetic
    # ``__user_<id>__`` role in the active workspace. Cascade handlers
    # (``set_can_manage_*_permission``, ``delete_*_permissions_cascade``,
    # ``rename_registered_model_permission``) call these directly. Per-resource
    # CRUD methods below remain only as tombstones backing the deprecated REST
    # surface; remove them when the deprecated handlers are dropped.

    def grant_user_permission(
        self,
        username: str,
        resource_type: str,
        resource_pattern: str,
        permission: str,
    ) -> None:
        """
        Upsert a ``permission`` grant on ``(resource_type, resource_pattern)`` for
        ``username`` via their synthetic role in the active workspace.
        """
        # Grain is validated at every write boundary, not just the role API: a pattern the type
        # does not declare would be stored and then silently ignored by the fold, so an operator
        # would get a success for a per-id DENY that protects nothing.
        _validate_resource_pattern(resource_pattern, resource_type)
        _validate_permission_for_resource_type(permission, resource_type)
        with self.ManagedSessionMaker(read_only=False) as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            role = self._get_or_create_synthetic_user_role(session, user.id, workspace_name)
            existing = (
                session
                .query(SqlRolePermission)
                .filter(
                    SqlRolePermission.role_id == role.id,
                    SqlRolePermission.resource_type == resource_type,
                    SqlRolePermission.resource_pattern == resource_pattern,
                )
                .first()
            )
            if existing is None:
                session.add(
                    SqlRolePermission(
                        role_id=role.id,
                        resource_type=resource_type,
                        resource_pattern=resource_pattern,
                        permission=permission,
                    )
                )
            else:
                existing.permission = permission

    @staticmethod
    def _reject_workspace_resource_type(resource_type: str) -> None:
        # Defense in depth — closes the ``sender_is_admin()`` bypass on the
        # ``validate_can_manage_resource`` gate.
        if resource_type == RESOURCE_TYPE_WORKSPACE:
            raise MlflowException.invalid_parameter_value(
                "resource_type 'workspace' is not supported by the per-user permission "
                "convenience APIs. Use set_workspace_permission / "
                "delete_workspace_permission for workspace-wide grants."
            )

    def grant_user_resource_permission(
        self,
        username: str,
        resource_type: str,
        resource_pattern: str,
        permission: str,
    ) -> None:
        """Insert one grant on ``username``'s synthetic role in the active workspace;
        raises ``RESOURCE_ALREADY_EXISTS`` if a row exists (matches the legacy
        ``create_*_permission`` contract).
        """
        self._reject_workspace_resource_type(resource_type)
        _validate_resource_pattern(resource_pattern, resource_type)
        _validate_permission_for_resource_type(permission, resource_type)
        duplicate_message = (
            f"Permission for user={username} on "
            f"resource_type={resource_type}, resource_id={resource_pattern} already exists."
        )
        with self.ManagedSessionMaker(read_only=False) as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            role = self._get_or_create_synthetic_user_role(session, user.id, workspace_name)
            existing = (
                session
                .query(SqlRolePermission)
                .filter(
                    SqlRolePermission.role_id == role.id,
                    SqlRolePermission.resource_type == resource_type,
                    SqlRolePermission.resource_pattern == resource_pattern,
                )
                .first()
            )
            if existing is not None:
                raise MlflowException(duplicate_message, RESOURCE_ALREADY_EXISTS)
            try:
                with session.begin_nested():
                    session.add(
                        SqlRolePermission(
                            role_id=role.id,
                            resource_type=resource_type,
                            resource_pattern=resource_pattern,
                            permission=permission,
                        )
                    )
                    session.flush()
            except IntegrityError as e:
                # Concurrent create lost the unique-constraint race. Surface as
                # a clean RESOURCE_ALREADY_EXISTS instead of a 500.
                raise MlflowException(duplicate_message, RESOURCE_ALREADY_EXISTS) from e

    def revoke_user_resource_permission(
        self,
        username: str,
        resource_type: str,
        resource_pattern: str,
    ) -> None:
        """Remove one grant from ``username``'s synthetic role in the active workspace;
        raises ``RESOURCE_DOES_NOT_EXIST`` if no row matches (matches the legacy
        ``delete_*_permission`` contract).
        """
        self._reject_workspace_resource_type(resource_type)
        _validate_resource_type(resource_type)
        not_found_message = (
            f"Permission for user={username} on "
            f"resource_type={resource_type}, resource_id={resource_pattern} not found."
        )
        with self.ManagedSessionMaker(read_only=False) as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            role = (
                session
                .query(SqlRole)
                .filter(
                    SqlRole.workspace == workspace_name,
                    SqlRole.name == self._synthetic_user_role_name(user.id),
                )
                .first()
            )
            rp = (
                None
                if role is None
                else session
                .query(SqlRolePermission)
                .filter(
                    SqlRolePermission.role_id == role.id,
                    SqlRolePermission.resource_type == resource_type,
                    SqlRolePermission.resource_pattern == resource_pattern,
                )
                .first()
            )
            if rp is None:
                raise MlflowException(not_found_message, RESOURCE_DOES_NOT_EXIST)
            session.delete(rp)

    def delete_grants_for_resource(
        self,
        resource_type: str,
        resource_pattern: str,
        *,
        workspace_scoped: bool = False,
    ) -> None:
        """
        Delete every synthetic-role grant matching ``(resource_type,
        resource_pattern)``. ``workspace_scoped=True`` restricts to the active
        workspace — for resources whose pattern can collide across workspaces
        (e.g. registered-model names). Admin-created roles are never touched.
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            workspace = self._get_active_workspace_name() if workspace_scoped else None
            role_ids = self._synthetic_role_ids(session, workspace=workspace)
            if not role_ids:
                return
            session.query(SqlRolePermission).filter(
                SqlRolePermission.role_id.in_(role_ids),
                SqlRolePermission.resource_type == resource_type,
                SqlRolePermission.resource_pattern == resource_pattern,
            ).delete(synchronize_session=False)

    def rename_grants_for_resource(
        self,
        resource_type: str,
        old_pattern: str,
        new_pattern: str,
        *,
        workspace_scoped: bool = False,
    ) -> None:
        """
        Rewrite every synthetic-role grant on ``(resource_type, old_pattern)`` to
        ``(resource_type, new_pattern)``. Used for resources whose pattern is the
        primary key and can change (e.g. registered-model rename).
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            workspace = self._get_active_workspace_name() if workspace_scoped else None
            role_ids = self._synthetic_role_ids(session, workspace=workspace)
            if not role_ids:
                return
            session.query(SqlRolePermission).filter(
                SqlRolePermission.role_id.in_(role_ids),
                SqlRolePermission.resource_type == resource_type,
                SqlRolePermission.resource_pattern == old_pattern,
            ).update(
                {SqlRolePermission.resource_pattern: new_pattern},
                synchronize_session=False,
            )

    def rename_conditions_for_registry_resource(self, old_name: str, new_name: str) -> None:
        """Follow a registered-model/prompt rename with the conditions scoped to that name.

        The companion to :meth:`rename_grants_for_resource`, and necessary for the same
        reason: a registry resource IS its name, so a rename moves the identity every
        scoped row is addressed by. Grants were already migrated; conditions were not, so
        a rename silently dropped every restriction on the resource while leaving the
        grants that the restrictions narrowed fully intact. That is the fail-open
        direction, and it is reachable by anyone who can rename.

        Two kinds of row are addressed by the old name, because the two scope axes carry
        it differently (see :func:`normalize_condition_scope`):

        - the **parent** rows -- type ``registered_model``/``prompt``, whose
          ``resource_pattern`` is the name itself;
        - the **version** rows -- type ``registered_model_version``/``prompt_version``,
          which are wildcard-only on their own axis and name the model as their
          ``container_resource_pattern`` instead.

        Both families are swept unconditionally, following the grant hook: names are unique
        across the registry, so exactly one matches and the other is a no-op -- cheaper and
        more robust than classifying the family from the response. Each version type is
        paired with its own container type, which is defence in depth rather than what
        provides the isolation: the per-type ``resource_type ==`` filter already partitions
        the rows, and :func:`normalize_condition_scope` refuses to store a row whose
        container is not its type's declared one. It costs nothing and keeps a corrupt row
        from being retargeted.

        Scoped to the active workspace, like the grant rename: a name identifies a
        different resource in a different workspace, so rewriting beyond it would retarget
        conditions at resources the rename never touched.

        Unlike the grant rename this is NOT limited to synthetic roles. Grants on named
        roles are managed by an admin through the role APIs, but conditions live on named
        roles by design -- that is the only place an admin can write one -- so limiting
        this to synthetic roles would miss essentially every condition.

        One transaction, so a rename cannot leave the parent row migrated and its versions
        stranded. No unique constraint can be violated: uniqueness is on
        ``(role_id, resource_type, condition_slot)``, and two rows scoped to the same name
        are harmless anyway because every applicable condition must pass.
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            workspace = self._get_active_workspace_name()
            role_ids = [
                role_id
                for (role_id,) in session
                .query(SqlRole.id)
                .filter(SqlRole.workspace == workspace)
                .all()
            ]
            if not role_ids:
                return
            session.query(SqlMutationConditions).filter(
                SqlMutationConditions.role_id.in_(role_ids),
                SqlMutationConditions.resource_type.in_((
                    RESOURCE_TYPE_REGISTERED_MODEL,
                    RESOURCE_TYPE_PROMPT,
                )),
                SqlMutationConditions.resource_pattern == old_name,
            ).update(
                {SqlMutationConditions.resource_pattern: new_name},
                synchronize_session=False,
            )
            for version_type, container_type in (
                (RESOURCE_TYPE_REGISTERED_MODEL_VERSION, RESOURCE_TYPE_REGISTERED_MODEL),
                (RESOURCE_TYPE_PROMPT_VERSION, RESOURCE_TYPE_PROMPT),
            ):
                session.query(SqlMutationConditions).filter(
                    SqlMutationConditions.role_id.in_(role_ids),
                    SqlMutationConditions.resource_type == version_type,
                    SqlMutationConditions.container_resource_type == container_type,
                    SqlMutationConditions.container_resource_pattern == old_name,
                ).update(
                    {SqlMutationConditions.container_resource_pattern: new_name},
                    synchronize_session=False,
                )

    # ---- Legacy per-resource CRUD (tombstones) ----
    #
    # The methods below back the deprecated REST handlers in ``__init__.py``.
    # Internal callers should use ``grant_user_permission`` /
    # ``delete_grants_for_resource`` / ``rename_grants_for_resource`` instead.
    # Remove these when the deprecated REST surface is dropped.

    def create_experiment_permission(
        self, experiment_id: str, username: str, permission: str
    ) -> ExperimentPermission:
        _validate_permission_for_resource_type(permission, RESOURCE_TYPE_EXPERIMENT)
        with self.ManagedSessionMaker(read_only=False) as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            role = self._get_or_create_synthetic_user_role(session, user.id, workspace_name)
            existing = (
                session
                .query(SqlRolePermission)
                .filter(
                    SqlRolePermission.role_id == role.id,
                    SqlRolePermission.resource_type == RESOURCE_TYPE_EXPERIMENT,
                    SqlRolePermission.resource_pattern == experiment_id,
                )
                .first()
            )
            duplicate_message = (
                f"Experiment permission (experiment_id={experiment_id}, "
                f"username={username}) already exists."
            )
            if existing is not None:
                raise MlflowException(duplicate_message, RESOURCE_ALREADY_EXISTS)
            try:
                with session.begin_nested():
                    session.add(
                        SqlRolePermission(
                            role_id=role.id,
                            resource_type=RESOURCE_TYPE_EXPERIMENT,
                            resource_pattern=experiment_id,
                            permission=permission,
                        )
                    )
                    session.flush()
            except IntegrityError as e:
                # Concurrent create lost the unique-constraint race. Surface as
                # a clean RESOURCE_ALREADY_EXISTS instead of a 500.
                raise MlflowException(duplicate_message, RESOURCE_ALREADY_EXISTS) from e
            return ExperimentPermission(
                experiment_id=experiment_id, user_id=user.id, permission=permission
            )

    def _get_experiment_permission_row(
        self, session, experiment_id: str, username: str
    ) -> tuple[SqlUser, SqlRolePermission]:
        user = self._get_user(session, username=username)
        workspace_name = self._get_active_workspace_name()
        role = (
            session
            .query(SqlRole)
            .filter(
                SqlRole.workspace == workspace_name,
                SqlRole.name == self._synthetic_user_role_name(user.id),
            )
            .first()
        )
        rp = (
            None
            if role is None
            else session
            .query(SqlRolePermission)
            .filter(
                SqlRolePermission.role_id == role.id,
                SqlRolePermission.resource_type == RESOURCE_TYPE_EXPERIMENT,
                SqlRolePermission.resource_pattern == experiment_id,
            )
            .first()
        )
        if rp is None:
            raise MlflowException(
                f"Experiment permission with experiment_id={experiment_id} and "
                f"username={username} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        return user, rp

    def get_experiment_permission(self, experiment_id: str, username: str) -> ExperimentPermission:
        with self.ManagedSessionMaker() as session:
            user, rp = self._get_experiment_permission_row(session, experiment_id, username)
            return ExperimentPermission(
                experiment_id=experiment_id, user_id=user.id, permission=rp.permission
            )

    def list_experiment_permissions(self, username: str) -> list[ExperimentPermission]:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            rows = (
                session
                .query(SqlRolePermission.resource_pattern, SqlRolePermission.permission)
                .join(SqlRole, SqlRole.id == SqlRolePermission.role_id)
                .filter(
                    SqlRole.name == self._synthetic_user_role_name(user.id),
                    SqlRolePermission.resource_type == RESOURCE_TYPE_EXPERIMENT,
                )
                .all()
            )
            return [
                ExperimentPermission(experiment_id=pattern, user_id=user.id, permission=permission)
                for pattern, permission in rows
            ]

    def update_experiment_permission(
        self, experiment_id: str, username: str, permission: str
    ) -> ExperimentPermission:
        _validate_permission_for_resource_type(permission, RESOURCE_TYPE_EXPERIMENT)
        with self.ManagedSessionMaker(read_only=False) as session:
            user, rp = self._get_experiment_permission_row(session, experiment_id, username)
            rp.permission = permission
            return ExperimentPermission(
                experiment_id=experiment_id, user_id=user.id, permission=permission
            )

    def delete_experiment_permission(self, experiment_id: str, username: str):
        with self.ManagedSessionMaker(read_only=False) as session:
            _, rp = self._get_experiment_permission_row(session, experiment_id, username)
            session.delete(rp)

    def delete_workspace_permissions_for_workspace(self, workspace_name: str) -> None:
        with self.ManagedSessionMaker(read_only=False) as session:
            role_ids = self._synthetic_role_ids(session, workspace=workspace_name)
            if not role_ids:
                return
            session.query(SqlRolePermission).filter(
                SqlRolePermission.role_id.in_(role_ids),
                SqlRolePermission.resource_type == RESOURCE_TYPE_WORKSPACE,
                SqlRolePermission.resource_pattern == "*",
            ).delete(synchronize_session=False)

    def create_registered_model_permission(
        self, name: str, username: str, permission: str
    ) -> RegisteredModelPermission:
        _validate_permission_for_resource_type(permission, RESOURCE_TYPE_REGISTERED_MODEL)
        with self.ManagedSessionMaker(read_only=False) as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            role = self._get_or_create_synthetic_user_role(session, user.id, workspace_name)
            existing = (
                session
                .query(SqlRolePermission)
                .filter(
                    SqlRolePermission.role_id == role.id,
                    SqlRolePermission.resource_type == RESOURCE_TYPE_REGISTERED_MODEL,
                    SqlRolePermission.resource_pattern == name,
                )
                .first()
            )
            duplicate_message = (
                "Registered model permission "
                f"with workspace={workspace_name}, name={name} and username={username} "
                "already exists."
            )
            if existing is not None:
                raise MlflowException(duplicate_message, RESOURCE_ALREADY_EXISTS)
            try:
                with session.begin_nested():
                    session.add(
                        SqlRolePermission(
                            role_id=role.id,
                            resource_type=RESOURCE_TYPE_REGISTERED_MODEL,
                            resource_pattern=name,
                            permission=permission,
                        )
                    )
                    session.flush()
            except IntegrityError as e:
                # Concurrent create lost the unique-constraint race. Surface as
                # a clean RESOURCE_ALREADY_EXISTS instead of a 500.
                raise MlflowException(duplicate_message, RESOURCE_ALREADY_EXISTS) from e
            return RegisteredModelPermission(
                workspace=workspace_name, name=name, user_id=user.id, permission=permission
            )

    def _get_registered_model_permission_row(
        self, session, name: str, username: str
    ) -> tuple[SqlUser, str, SqlRolePermission]:
        user = self._get_user(session, username=username)
        workspace_name = self._get_active_workspace_name()
        role = (
            session
            .query(SqlRole)
            .filter(
                SqlRole.workspace == workspace_name,
                SqlRole.name == self._synthetic_user_role_name(user.id),
            )
            .first()
        )
        rp = (
            None
            if role is None
            else session
            .query(SqlRolePermission)
            .filter(
                SqlRolePermission.role_id == role.id,
                SqlRolePermission.resource_type == RESOURCE_TYPE_REGISTERED_MODEL,
                SqlRolePermission.resource_pattern == name,
            )
            .first()
        )
        if rp is None:
            raise MlflowException(
                "Registered model permission "
                f"with workspace={workspace_name}, name={name} and username={username} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        return user, workspace_name, rp

    def get_registered_model_permission(
        self, name: str, username: str
    ) -> RegisteredModelPermission:
        with self.ManagedSessionMaker() as session:
            user, workspace_name, rp = self._get_registered_model_permission_row(
                session, name, username
            )
            return RegisteredModelPermission(
                workspace=workspace_name, name=name, user_id=user.id, permission=rp.permission
            )

    def list_registered_model_permissions(self, username: str) -> list[RegisteredModelPermission]:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            rows = (
                session
                .query(SqlRolePermission.resource_pattern, SqlRolePermission.permission)
                .join(SqlRole, SqlRole.id == SqlRolePermission.role_id)
                .filter(
                    SqlRole.workspace == workspace_name,
                    SqlRole.name == self._synthetic_user_role_name(user.id),
                    SqlRolePermission.resource_type == RESOURCE_TYPE_REGISTERED_MODEL,
                )
                .all()
            )
            return [
                RegisteredModelPermission(
                    workspace=workspace_name,
                    name=pattern,
                    user_id=user.id,
                    permission=permission,
                )
                for pattern, permission in rows
            ]

    def list_all_registered_model_permissions(
        self, username: str
    ) -> list[RegisteredModelPermission]:
        """
        Cross-workspace variant for callers without an active workspace
        (e.g. the global ``/users/current/permissions`` endpoint backing
        the ``/account`` page). Mirrors ``list_registered_model_permissions``
        but skips the workspace filter so the returned rows span every
        workspace the user has grants in — each row carries its own
        ``workspace`` value (taken from the synthetic role's workspace),
        so the caller can still attribute correctly.
        """
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            rows = (
                session
                .query(
                    SqlRole.workspace,
                    SqlRolePermission.resource_pattern,
                    SqlRolePermission.permission,
                )
                .join(SqlRolePermission, SqlRole.id == SqlRolePermission.role_id)
                .filter(
                    SqlRole.name == self._synthetic_user_role_name(user.id),
                    SqlRolePermission.resource_type == RESOURCE_TYPE_REGISTERED_MODEL,
                )
                .all()
            )
            return [
                RegisteredModelPermission(
                    workspace=workspace,
                    name=pattern,
                    user_id=user.id,
                    permission=permission,
                )
                for workspace, pattern, permission in rows
            ]

    def update_registered_model_permission(
        self, name: str, username: str, permission: str
    ) -> RegisteredModelPermission:
        _validate_permission_for_resource_type(permission, RESOURCE_TYPE_REGISTERED_MODEL)
        with self.ManagedSessionMaker(read_only=False) as session:
            user, workspace_name, rp = self._get_registered_model_permission_row(
                session, name, username
            )
            rp.permission = permission
            return RegisteredModelPermission(
                workspace=workspace_name, name=name, user_id=user.id, permission=permission
            )

    def delete_registered_model_permission(self, name: str, username: str):
        with self.ManagedSessionMaker(read_only=False) as session:
            _, _, rp = self._get_registered_model_permission_row(session, name, username)
            session.delete(rp)

    def _list_workspace_perm_rows(
        self,
        session,
        *,
        workspace_name: str | None = None,
        user_id: int | None = None,
    ) -> list[tuple[SqlRole, SqlRolePermission]]:
        query = (
            session
            .query(SqlRole, SqlRolePermission)
            .join(SqlRolePermission, SqlRolePermission.role_id == SqlRole.id)
            .filter(
                SqlRolePermission.resource_type == RESOURCE_TYPE_WORKSPACE,
                SqlRolePermission.resource_pattern == "*",
            )
        )
        if workspace_name is not None:
            query = query.filter(SqlRole.workspace == workspace_name)
        if user_id is not None:
            query = query.join(
                SqlUserRoleAssignment, SqlUserRoleAssignment.role_id == SqlRole.id
            ).filter(SqlUserRoleAssignment.user_id == user_id)
        return [(role, rp) for (role, rp) in query.all() if self._is_synthetic_role_name(role.name)]

    @classmethod
    def _user_id_from_synthetic_role_name(cls, name: str) -> int:
        return int(name[len(cls._SYNTHETIC_ROLE_PREFIX) : -len(cls._SYNTHETIC_ROLE_SUFFIX)])

    def list_workspace_permissions(self, workspace_name: str) -> list[WorkspacePermission]:
        with self.ManagedSessionMaker() as session:
            rows = self._list_workspace_perm_rows(session, workspace_name=workspace_name)
            return [
                WorkspacePermission(
                    workspace=role.workspace,
                    user_id=self._user_id_from_synthetic_role_name(role.name),
                    permission=rp.permission,
                )
                for (role, rp) in rows
            ]

    def list_user_workspace_permissions(self, username: str) -> list[WorkspacePermission]:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            rows = self._list_workspace_perm_rows(session, user_id=user.id)
            return [
                WorkspacePermission(
                    workspace=role.workspace,
                    user_id=user.id,
                    permission=rp.permission,
                )
                for (role, rp) in rows
            ]

    def set_workspace_permission(
        self, workspace_name: str, username: str, permission: str
    ) -> WorkspacePermission:
        _validate_permission_for_resource_type(permission, RESOURCE_TYPE_WORKSPACE)
        with self.ManagedSessionMaker(read_only=False) as session:
            user = self._get_user(session, username=username)
            role = self._get_or_create_synthetic_user_role(session, user.id, workspace_name)
            existing = (
                session
                .query(SqlRolePermission)
                .filter(
                    SqlRolePermission.role_id == role.id,
                    SqlRolePermission.resource_type == RESOURCE_TYPE_WORKSPACE,
                    SqlRolePermission.resource_pattern == "*",
                )
                .first()
            )
            if existing is None:
                session.add(
                    SqlRolePermission(
                        role_id=role.id,
                        resource_type=RESOURCE_TYPE_WORKSPACE,
                        resource_pattern="*",
                        permission=permission,
                    )
                )
            else:
                existing.permission = permission
            session.flush()
            return WorkspacePermission(
                workspace=workspace_name, user_id=user.id, permission=permission
            )

    def delete_workspace_permission(self, workspace_name: str, username: str) -> None:
        with self.ManagedSessionMaker(read_only=False) as session:
            user = self._get_user(session, username=username)
            role_name = self._synthetic_user_role_name(user.id)
            role = (
                session
                .query(SqlRole)
                .filter(SqlRole.workspace == workspace_name, SqlRole.name == role_name)
                .first()
            )
            existing = None
            if role is not None:
                existing = (
                    session
                    .query(SqlRolePermission)
                    .filter(
                        SqlRolePermission.role_id == role.id,
                        SqlRolePermission.resource_type == RESOURCE_TYPE_WORKSPACE,
                        SqlRolePermission.resource_pattern == "*",
                    )
                    .first()
                )
            if existing is None:
                raise MlflowException(
                    (
                        "Workspace permission does not exist for "
                        f"workspace='{workspace_name}', username='{username}'"
                    ),
                    RESOURCE_DOES_NOT_EXIST,
                )
            session.delete(existing)

    def list_accessible_workspace_names(self, username: str | None) -> set[str]:
        """
        Return the set of workspaces ``username`` can see. Two-source union that
        mirrors master's #22864 logic, with the legacy half adapted to the
        post-migration storage shape:

        - **Non-synthetic role assignments**: any role the user is assigned to
          (admin-created, no permissions or otherwise) confers visibility on
          its workspace. Matches #22864's "any role assignment counts".
        - **Synthetic ``__user_<id>__`` roles**: only confer visibility if their
          ``('workspace', '*')`` grant has ``can_read``. This preserves the legacy
          ``workspace_permissions`` semantic where a row with
          ``permission='NO_PERMISSIONS'`` hides the workspace; those rows now
          live as the ``('workspace', '*')`` grant on the synthetic role.
        """
        if username is None:
            return set()
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            # All workspace/role-name pairs the user is assigned to.
            assigned = (
                session
                .query(SqlRole.workspace, SqlRole.name)
                .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
                .filter(SqlUserRoleAssignment.user_id == user.id)
                .distinct()
                .all()
            )
            accessible: set[str] = set()
            synthetic_workspaces: set[str] = set()
            for ws, role_name in assigned:
                if self._is_synthetic_role_name(role_name):
                    synthetic_workspaces.add(ws)
                else:
                    accessible.add(ws)

            if synthetic_workspaces:
                # For synthetic roles, only count workspaces where the (*, *) grant
                # has can_read. A migrated NO_PERMISSIONS row keeps the workspace
                # hidden.
                synthetic_rows = (
                    session
                    .query(SqlRole.workspace, SqlRolePermission.permission)
                    .join(SqlRolePermission, SqlRolePermission.role_id == SqlRole.id)
                    .join(SqlUserRoleAssignment, SqlUserRoleAssignment.role_id == SqlRole.id)
                    .filter(
                        SqlUserRoleAssignment.user_id == user.id,
                        SqlRole.workspace.in_(synthetic_workspaces),
                        SqlRolePermission.resource_type == RESOURCE_TYPE_WORKSPACE,
                        SqlRolePermission.resource_pattern == "*",
                    )
                    .all()
                )
                accessible.update(
                    ws for (ws, perm) in synthetic_rows if get_permission(perm).can_read
                )
            return accessible

    def get_workspace_permission(self, workspace_name: str, username: str) -> Permission | None:
        """
        Get the user's **direct** workspace permission — the
        ``('workspace', '*', PERMISSION)`` grant on their synthetic
        ``__user_<id>__`` role for ``workspace_name``, if any. Pre-migration this
        was a row in the legacy ``workspace_permissions`` table; the
        ``e5f6a7b8c9d0`` migration moved it to ``role_permissions`` under the
        synthetic role, which is what this method now queries.

        Does NOT include grants conferred by other (admin-managed) roles.
        Callers that need the full authorization picture should also consult
        ``get_role_workspace_permission`` and ``max_permission``-merge the two.
        See ``mlflow.server.auth.__init__._workspace_permission`` for the
        canonical aggregation.
        """
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            role_name = self._synthetic_user_role_name(user.id)
            rp = (
                session
                .query(SqlRolePermission)
                .join(SqlRole, SqlRole.id == SqlRolePermission.role_id)
                .filter(
                    SqlRole.workspace == workspace_name,
                    SqlRole.name == role_name,
                    SqlRolePermission.resource_type == RESOURCE_TYPE_WORKSPACE,
                    SqlRolePermission.resource_pattern == "*",
                )
                .first()
            )
            if rp is not None:
                return get_permission(rp.permission)
        return None

    def get_role_workspace_permission(
        self, workspace_name: str, username: str
    ) -> Permission | None:
        """
        Highest **role-based** permission ``username`` has on ``workspace_name``
        where the role grant is workspace-wide. Returns ``None`` when there are
        no such grants.

        The simplified model uses a single workspace-wide grant slot,
        ``(resource_type='workspace', resource_pattern='*')``. The permission
        tier (USE for regular member, MANAGE for workspace admin) carries the
        admin signal — both seeded roles (``user`` and ``admin``) and migrated
        legacy ``workspace_permissions`` rows land in this slot.

        Complements ``get_workspace_permission`` — that helper restricts to the
        user's own synthetic ``__user_<id>__`` role, this one walks every role
        the user is assigned to (admin-managed and synthetic alike). Callers
        that need the effective workspace-level permission should max-merge the
        two.
        """
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            permissions = (
                session
                .query(SqlRolePermission.permission)
                .join(SqlRole, SqlRole.id == SqlRolePermission.role_id)
                .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
                .filter(
                    SqlUserRoleAssignment.user_id == user.id,
                    SqlRole.workspace == workspace_name,
                    SqlRolePermission.resource_type == RESOURCE_TYPE_WORKSPACE,
                    SqlRolePermission.resource_pattern == "*",
                )
                .distinct()
                .all()
            )
        if not permissions:
            return None
        best: str | None = None
        for (perm,) in permissions:
            best = perm if best is None else max_permission(best, perm)
        return get_permission(best)

    @staticmethod
    def _scorer_pattern(experiment_id: str, scorer_name: str) -> str:
        # Scorer names may contain arbitrary characters including ``/`` (see
        # ``validate_scorer_name``, which only forbids empty/whitespace). We
        # URL-encode the name component so the pattern ``<experiment_id>/<name>``
        # is unambiguous; the migration (``e5f6a7b8c9d0``) uses the same
        # encoding, so post-migration and live grants line up.
        return f"{experiment_id}/{quote(scorer_name, safe='')}"

    def create_scorer_permission(
        self, experiment_id: str, scorer_name: str, username: str, permission: str
    ) -> ScorerPermission:
        _validate_permission_for_resource_type(permission, RESOURCE_TYPE_SCORER)
        pattern = self._scorer_pattern(experiment_id, scorer_name)
        with self.ManagedSessionMaker(read_only=False) as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            role = self._get_or_create_synthetic_user_role(session, user.id, workspace_name)
            existing = (
                session
                .query(SqlRolePermission)
                .filter(
                    SqlRolePermission.role_id == role.id,
                    SqlRolePermission.resource_type == RESOURCE_TYPE_SCORER,
                    SqlRolePermission.resource_pattern == pattern,
                )
                .first()
            )
            duplicate_message = (
                f"Scorer permission (experiment_id={experiment_id}, "
                f"scorer_name={scorer_name}, username={username}) already exists."
            )
            if existing is not None:
                raise MlflowException(duplicate_message, RESOURCE_ALREADY_EXISTS)
            try:
                with session.begin_nested():
                    session.add(
                        SqlRolePermission(
                            role_id=role.id,
                            resource_type=RESOURCE_TYPE_SCORER,
                            resource_pattern=pattern,
                            permission=permission,
                        )
                    )
                    session.flush()
            except IntegrityError as e:
                # Concurrent create lost the unique-constraint race. Surface as
                # a clean RESOURCE_ALREADY_EXISTS instead of a 500.
                raise MlflowException(duplicate_message, RESOURCE_ALREADY_EXISTS) from e
            return ScorerPermission(
                experiment_id=experiment_id,
                scorer_name=scorer_name,
                user_id=user.id,
                permission=permission,
            )

    def _get_scorer_permission_row(
        self, session, experiment_id: str, scorer_name: str, username: str
    ) -> tuple[SqlUser, SqlRolePermission]:
        user = self._get_user(session, username=username)
        workspace_name = self._get_active_workspace_name()
        role = (
            session
            .query(SqlRole)
            .filter(
                SqlRole.workspace == workspace_name,
                SqlRole.name == self._synthetic_user_role_name(user.id),
            )
            .first()
        )
        pattern = self._scorer_pattern(experiment_id, scorer_name)
        rp = (
            None
            if role is None
            else session
            .query(SqlRolePermission)
            .filter(
                SqlRolePermission.role_id == role.id,
                SqlRolePermission.resource_type == RESOURCE_TYPE_SCORER,
                SqlRolePermission.resource_pattern == pattern,
            )
            .first()
        )
        if rp is None:
            raise MlflowException(
                f"Scorer permission with experiment_id={experiment_id}, "
                f"scorer_name={scorer_name}, and username={username} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        return user, rp

    def get_scorer_permission(
        self, experiment_id: str, scorer_name: str, username: str
    ) -> ScorerPermission:
        with self.ManagedSessionMaker() as session:
            user, rp = self._get_scorer_permission_row(
                session, experiment_id, scorer_name, username
            )
            return ScorerPermission(
                experiment_id=experiment_id,
                scorer_name=scorer_name,
                user_id=user.id,
                permission=rp.permission,
            )

    def list_scorer_permissions(self, username: str) -> list[ScorerPermission]:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            rows = (
                session
                .query(SqlRolePermission.resource_pattern, SqlRolePermission.permission)
                .join(SqlRole, SqlRole.id == SqlRolePermission.role_id)
                .filter(
                    SqlRole.name == self._synthetic_user_role_name(user.id),
                    SqlRolePermission.resource_type == RESOURCE_TYPE_SCORER,
                )
                .all()
            )
            out = []
            for pattern, permission in rows:
                # Compound key encoded as ``{experiment_id}/{url_quote(scorer_name)}``
                # — see ``_scorer_pattern``. The first ``/`` is the delimiter; the
                # scorer name is URL-decoded back to its raw form.
                exp_id, _, sname_encoded = pattern.partition("/")
                sname = unquote(sname_encoded)
                out.append(
                    ScorerPermission(
                        experiment_id=exp_id,
                        scorer_name=sname,
                        user_id=user.id,
                        permission=permission,
                    )
                )
            return out

    def update_scorer_permission(
        self, experiment_id: str, scorer_name: str, username: str, permission: str
    ) -> ScorerPermission:
        _validate_permission_for_resource_type(permission, RESOURCE_TYPE_SCORER)
        with self.ManagedSessionMaker(read_only=False) as session:
            user, rp = self._get_scorer_permission_row(
                session, experiment_id, scorer_name, username
            )
            rp.permission = permission
            return ScorerPermission(
                experiment_id=experiment_id,
                scorer_name=scorer_name,
                user_id=user.id,
                permission=permission,
            )

    def delete_scorer_permission(self, experiment_id: str, scorer_name: str, username: str):
        with self.ManagedSessionMaker(read_only=False) as session:
            _, rp = self._get_scorer_permission_row(session, experiment_id, scorer_name, username)
            session.delete(rp)

    # ---- Gateway permission helpers ----
    #
    # The three gateway resource types (secret / endpoint / model_definition) share a
    # common shape: one resource id, per-user permission, no workspace column. These
    # helpers factor out the synthetic-role plumbing so the public methods stay a
    # thin wrapper with type-specific entity construction.

    def _create_per_resource_permission(
        self,
        *,
        resource_type: str,
        resource_pattern: str,
        username: str,
        permission: str,
        entity_factory,
        duplicate_message: str,
    ):
        _validate_permission_for_resource_type(permission, resource_type)
        with self.ManagedSessionMaker(read_only=False) as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            role = self._get_or_create_synthetic_user_role(session, user.id, workspace_name)
            existing = (
                session
                .query(SqlRolePermission)
                .filter(
                    SqlRolePermission.role_id == role.id,
                    SqlRolePermission.resource_type == resource_type,
                    SqlRolePermission.resource_pattern == resource_pattern,
                )
                .first()
            )
            if existing is not None:
                raise MlflowException(duplicate_message, RESOURCE_ALREADY_EXISTS)
            try:
                with session.begin_nested():
                    session.add(
                        SqlRolePermission(
                            role_id=role.id,
                            resource_type=resource_type,
                            resource_pattern=resource_pattern,
                            permission=permission,
                        )
                    )
                    session.flush()
            except IntegrityError as e:
                # Concurrent create lost the unique-constraint race. Surface as
                # a clean RESOURCE_ALREADY_EXISTS instead of a 500.
                raise MlflowException(duplicate_message, RESOURCE_ALREADY_EXISTS) from e
            return entity_factory(user=user, permission=permission)

    def _get_per_resource_permission_row(
        self,
        session,
        *,
        resource_type: str,
        resource_pattern: str,
        username: str,
        not_found_message: str,
    ) -> tuple[SqlUser, SqlRolePermission]:
        user = self._get_user(session, username=username)
        workspace_name = self._get_active_workspace_name()
        role = (
            session
            .query(SqlRole)
            .filter(
                SqlRole.workspace == workspace_name,
                SqlRole.name == self._synthetic_user_role_name(user.id),
            )
            .first()
        )
        rp = (
            None
            if role is None
            else session
            .query(SqlRolePermission)
            .filter(
                SqlRolePermission.role_id == role.id,
                SqlRolePermission.resource_type == resource_type,
                SqlRolePermission.resource_pattern == resource_pattern,
            )
            .first()
        )
        if rp is None:
            raise MlflowException(not_found_message, RESOURCE_DOES_NOT_EXIST)
        return user, rp

    def _update_per_resource_permission(
        self,
        *,
        resource_type: str,
        resource_pattern: str,
        username: str,
        permission: str,
        entity_factory,
        not_found_message: str,
    ):
        _validate_permission_for_resource_type(permission, resource_type)
        with self.ManagedSessionMaker(read_only=False) as session:
            user, rp = self._get_per_resource_permission_row(
                session,
                resource_type=resource_type,
                resource_pattern=resource_pattern,
                username=username,
                not_found_message=not_found_message,
            )
            rp.permission = permission
            return entity_factory(user=user, permission=permission)

    def _delete_per_resource_permission(
        self,
        *,
        resource_type: str,
        resource_pattern: str,
        username: str,
        not_found_message: str,
    ) -> None:
        with self.ManagedSessionMaker(read_only=False) as session:
            _, rp = self._get_per_resource_permission_row(
                session,
                resource_type=resource_type,
                resource_pattern=resource_pattern,
                username=username,
                not_found_message=not_found_message,
            )
            session.delete(rp)

    def _list_per_resource_permissions(self, username: str, resource_type: str):
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            user_id = user.id
            workspace_name = self._get_active_workspace_name()
            rows = (
                session
                .query(SqlRolePermission.resource_pattern, SqlRolePermission.permission)
                .join(SqlRole, SqlRole.id == SqlRolePermission.role_id)
                .filter(
                    SqlRole.workspace == workspace_name,
                    SqlRole.name == self._synthetic_user_role_name(user_id),
                    SqlRolePermission.resource_type == resource_type,
                )
                .all()
            )
            # Read ``user.id`` while the session is still open and return the
            # plain int — callers that build entity objects outside the session
            # block would otherwise hit ``DetachedInstanceError`` on attribute
            # access.
            return user_id, rows

    # ---- gateway_secret ----

    def create_gateway_secret_permission(
        self, secret_id: str, username: str, permission: str
    ) -> GatewaySecretPermission:
        return self._create_per_resource_permission(
            resource_type=RESOURCE_TYPE_GATEWAY_SECRET,
            resource_pattern=secret_id,
            username=username,
            permission=permission,
            entity_factory=lambda user, permission: GatewaySecretPermission(
                secret_id=secret_id, user_id=user.id, permission=permission
            ),
            duplicate_message=(
                f"Gateway secret permission (secret_id={secret_id}, username={username}) "
                "already exists."
            ),
        )

    def get_gateway_secret_permission(
        self, secret_id: str, username: str
    ) -> GatewaySecretPermission:
        with self.ManagedSessionMaker() as session:
            user, rp = self._get_per_resource_permission_row(
                session,
                resource_type=RESOURCE_TYPE_GATEWAY_SECRET,
                resource_pattern=secret_id,
                username=username,
                not_found_message=(
                    f"Gateway secret permission with secret_id={secret_id} and "
                    f"username={username} not found"
                ),
            )
            return GatewaySecretPermission(
                secret_id=secret_id, user_id=user.id, permission=rp.permission
            )

    def list_gateway_secret_permissions(self, username: str) -> list[GatewaySecretPermission]:
        user_id, rows = self._list_per_resource_permissions(username, RESOURCE_TYPE_GATEWAY_SECRET)
        return [
            GatewaySecretPermission(secret_id=p, user_id=user_id, permission=perm)
            for p, perm in rows
        ]

    def update_gateway_secret_permission(
        self, secret_id: str, username: str, permission: str
    ) -> GatewaySecretPermission:
        return self._update_per_resource_permission(
            resource_type=RESOURCE_TYPE_GATEWAY_SECRET,
            resource_pattern=secret_id,
            username=username,
            permission=permission,
            entity_factory=lambda user, permission: GatewaySecretPermission(
                secret_id=secret_id, user_id=user.id, permission=permission
            ),
            not_found_message=(
                f"Gateway secret permission with secret_id={secret_id} and "
                f"username={username} not found"
            ),
        )

    def delete_gateway_secret_permission(self, secret_id: str, username: str) -> None:
        self._delete_per_resource_permission(
            resource_type=RESOURCE_TYPE_GATEWAY_SECRET,
            resource_pattern=secret_id,
            username=username,
            not_found_message=(
                f"Gateway secret permission with secret_id={secret_id} and "
                f"username={username} not found"
            ),
        )

    # ---- gateway_endpoint ----

    def create_gateway_endpoint_permission(
        self, endpoint_id: str, username: str, permission: str
    ) -> GatewayEndpointPermission:
        return self._create_per_resource_permission(
            resource_type=RESOURCE_TYPE_GATEWAY_ENDPOINT,
            resource_pattern=endpoint_id,
            username=username,
            permission=permission,
            entity_factory=lambda user, permission: GatewayEndpointPermission(
                endpoint_id=endpoint_id, user_id=user.id, permission=permission
            ),
            duplicate_message=(
                f"Gateway endpoint permission (endpoint_id={endpoint_id}, "
                f"username={username}) already exists."
            ),
        )

    def get_gateway_endpoint_permission(
        self, endpoint_id: str, username: str
    ) -> GatewayEndpointPermission:
        with self.ManagedSessionMaker() as session:
            user, rp = self._get_per_resource_permission_row(
                session,
                resource_type=RESOURCE_TYPE_GATEWAY_ENDPOINT,
                resource_pattern=endpoint_id,
                username=username,
                not_found_message=(
                    f"Gateway endpoint permission with endpoint_id={endpoint_id} and "
                    f"username={username} not found"
                ),
            )
            return GatewayEndpointPermission(
                endpoint_id=endpoint_id, user_id=user.id, permission=rp.permission
            )

    def list_gateway_endpoint_permissions(self, username: str) -> list[GatewayEndpointPermission]:
        user_id, rows = self._list_per_resource_permissions(
            username, RESOURCE_TYPE_GATEWAY_ENDPOINT
        )
        return [
            GatewayEndpointPermission(endpoint_id=p, user_id=user_id, permission=perm)
            for p, perm in rows
        ]

    def update_gateway_endpoint_permission(
        self, endpoint_id: str, username: str, permission: str
    ) -> GatewayEndpointPermission:
        return self._update_per_resource_permission(
            resource_type=RESOURCE_TYPE_GATEWAY_ENDPOINT,
            resource_pattern=endpoint_id,
            username=username,
            permission=permission,
            entity_factory=lambda user, permission: GatewayEndpointPermission(
                endpoint_id=endpoint_id, user_id=user.id, permission=permission
            ),
            not_found_message=(
                f"Gateway endpoint permission with endpoint_id={endpoint_id} and "
                f"username={username} not found"
            ),
        )

    def delete_gateway_endpoint_permission(self, endpoint_id: str, username: str) -> None:
        self._delete_per_resource_permission(
            resource_type=RESOURCE_TYPE_GATEWAY_ENDPOINT,
            resource_pattern=endpoint_id,
            username=username,
            not_found_message=(
                f"Gateway endpoint permission with endpoint_id={endpoint_id} and "
                f"username={username} not found"
            ),
        )

    # ---- gateway_model_definition ----

    def create_gateway_model_definition_permission(
        self, model_definition_id: str, username: str, permission: str
    ) -> GatewayModelDefinitionPermission:
        return self._create_per_resource_permission(
            resource_type=RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION,
            resource_pattern=model_definition_id,
            username=username,
            permission=permission,
            entity_factory=lambda user, permission: GatewayModelDefinitionPermission(
                model_definition_id=model_definition_id,
                user_id=user.id,
                permission=permission,
            ),
            duplicate_message=(
                f"Gateway model definition permission "
                f"(model_definition_id={model_definition_id}, username={username}) "
                "already exists."
            ),
        )

    def get_gateway_model_definition_permission(
        self, model_definition_id: str, username: str
    ) -> GatewayModelDefinitionPermission:
        with self.ManagedSessionMaker() as session:
            user, rp = self._get_per_resource_permission_row(
                session,
                resource_type=RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION,
                resource_pattern=model_definition_id,
                username=username,
                not_found_message=(
                    f"Gateway model definition permission with "
                    f"model_definition_id={model_definition_id} and "
                    f"username={username} not found"
                ),
            )
            return GatewayModelDefinitionPermission(
                model_definition_id=model_definition_id,
                user_id=user.id,
                permission=rp.permission,
            )

    def list_gateway_model_definition_permissions(
        self, username: str
    ) -> list[GatewayModelDefinitionPermission]:
        user_id, rows = self._list_per_resource_permissions(
            username, RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION
        )
        return [
            GatewayModelDefinitionPermission(
                model_definition_id=p, user_id=user_id, permission=perm
            )
            for p, perm in rows
        ]

    def update_gateway_model_definition_permission(
        self, model_definition_id: str, username: str, permission: str
    ) -> GatewayModelDefinitionPermission:
        return self._update_per_resource_permission(
            resource_type=RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION,
            resource_pattern=model_definition_id,
            username=username,
            permission=permission,
            entity_factory=lambda user, permission: GatewayModelDefinitionPermission(
                model_definition_id=model_definition_id,
                user_id=user.id,
                permission=permission,
            ),
            not_found_message=(
                f"Gateway model definition permission with "
                f"model_definition_id={model_definition_id} and username={username} not found"
            ),
        )

    def delete_gateway_model_definition_permission(
        self, model_definition_id: str, username: str
    ) -> None:
        self._delete_per_resource_permission(
            resource_type=RESOURCE_TYPE_GATEWAY_MODEL_DEFINITION,
            resource_pattern=model_definition_id,
            username=username,
            not_found_message=(
                f"Gateway model definition permission with "
                f"model_definition_id={model_definition_id} and username={username} not found"
            ),
        )

    # ---- mcp_server ----

    def create_mcp_server_permission(
        self, name: str, username: str, permission: str
    ) -> MCPServerPermission:
        return self._create_per_resource_permission(
            resource_type=RESOURCE_TYPE_MCP_SERVER,
            resource_pattern=name,
            username=username,
            permission=permission,
            entity_factory=lambda user, permission: MCPServerPermission(
                name=name, user_id=user.id, permission=permission
            ),
            duplicate_message=(
                f"MCP server permission (name={name}, username={username}) already exists."
            ),
        )

    def get_mcp_server_permission(self, name: str, username: str) -> MCPServerPermission:
        with self.ManagedSessionMaker() as session:
            user, rp = self._get_per_resource_permission_row(
                session,
                resource_type=RESOURCE_TYPE_MCP_SERVER,
                resource_pattern=name,
                username=username,
                not_found_message=(
                    f"MCP server permission with name={name} and username={username} not found"
                ),
            )
            return MCPServerPermission(name=name, user_id=user.id, permission=rp.permission)

    def list_mcp_server_permissions(self, username: str) -> list[MCPServerPermission]:
        user_id, rows = self._list_per_resource_permissions(username, RESOURCE_TYPE_MCP_SERVER)
        return [MCPServerPermission(name=p, user_id=user_id, permission=perm) for p, perm in rows]

    def update_mcp_server_permission(
        self, name: str, username: str, permission: str
    ) -> MCPServerPermission:
        return self._update_per_resource_permission(
            resource_type=RESOURCE_TYPE_MCP_SERVER,
            resource_pattern=name,
            username=username,
            permission=permission,
            entity_factory=lambda user, permission: MCPServerPermission(
                name=name, user_id=user.id, permission=permission
            ),
            not_found_message=(
                f"MCP server permission with name={name} and username={username} not found"
            ),
        )

    def delete_mcp_server_permission(self, name: str, username: str) -> None:
        self._delete_per_resource_permission(
            resource_type=RESOURCE_TYPE_MCP_SERVER,
            resource_pattern=name,
            username=username,
            not_found_message=(
                f"MCP server permission with name={name} and username={username} not found"
            ),
        )

    # ---- Role CRUD ----

    def create_role(
        self,
        name: str,
        workspace: str,
        description: str | None = None,
    ) -> Role:
        self._reject_synthetic_role_name(name)
        with self.ManagedSessionMaker(read_only=False) as session:
            try:
                role = SqlRole(
                    name=name,
                    workspace=workspace,
                    description=description,
                )
                session.add(role)
                session.flush()
                return role.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Role (name={name}, workspace={workspace}) already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                ) from e

    @staticmethod
    def _get_role(session, role_id: int) -> SqlRole:
        try:
            return session.query(SqlRole).filter(SqlRole.id == role_id).one()
        except NoResultFound:
            raise MlflowException(
                f"Role with id={role_id} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                f"Found multiple roles with id={role_id}",
                INVALID_STATE,
            )

    @staticmethod
    def _get_role_by_name(session, workspace: str, name: str) -> SqlRole:
        try:
            return (
                session
                .query(SqlRole)
                .filter(SqlRole.workspace == workspace, SqlRole.name == name)
                .one()
            )
        except NoResultFound:
            raise MlflowException(
                f"Role with name={name} in workspace={workspace} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                f"Found multiple roles with name={name} in workspace={workspace}",
                INVALID_STATE,
            )

    def get_role(self, role_id: int) -> Role:
        with self.ManagedSessionMaker() as session:
            return self._get_role(session, role_id).to_mlflow_entity()

    def get_role_by_name(self, workspace: str, name: str) -> Role:
        with self.ManagedSessionMaker() as session:
            return self._get_role_by_name(session, workspace, name).to_mlflow_entity()

    def list_roles(self, workspaces: Iterable[str] | None = None) -> list[Role]:
        # ``None`` lists every role across the system (admin path); an explicit
        # iterable scopes the listing to those workspaces. An empty iterable is
        # interpreted literally and returns no roles.
        if workspaces is None:
            with self.ManagedSessionMaker() as session:
                roles = session.query(SqlRole).options(selectinload(SqlRole.permissions)).all()
                return [r.to_mlflow_entity() for r in roles]
        names = list(workspaces)
        if not names:
            return []
        with self.ManagedSessionMaker() as session:
            roles = (
                session
                .query(SqlRole)
                .options(selectinload(SqlRole.permissions))
                .filter(SqlRole.workspace.in_(names))
                .all()
            )
            return [r.to_mlflow_entity() for r in roles]

    def update_role(
        self,
        role_id: int,
        name: str | None = None,
        description: str | None = None,
    ) -> Role:
        if name is not None:
            self._reject_synthetic_role_name(name)
        with self.ManagedSessionMaker(read_only=False) as session:
            role = self._get_role(session, role_id)
            if name is not None:
                # Check for name conflicts before updating
                existing = (
                    session
                    .query(SqlRole)
                    .filter(
                        SqlRole.workspace == role.workspace,
                        SqlRole.name == name,
                        SqlRole.id != role_id,
                    )
                    .first()
                )
                if existing is not None:
                    raise MlflowException(
                        f"Role with name={name} already exists in workspace={role.workspace}",
                        RESOURCE_ALREADY_EXISTS,
                    )
                role.name = name
            if description is not None:
                role.description = description
            return role.to_mlflow_entity()

    def delete_role(self, role_id: int) -> None:
        with self.ManagedSessionMaker(read_only=False) as session:
            role = self._get_role(session, role_id)
            session.delete(role)

    def delete_roles_for_workspace(self, workspace_name: str) -> None:
        # Batch delete: ORM-level ``cascade="all, delete-orphan"`` only fires when calling
        # ``session.delete(instance)``, so for a bulk delete we must explicitly remove
        # child rows (``role_permissions``, ``user_role_assignments``) before the roles
        # themselves. The FK doesn't declare ``ON DELETE CASCADE`` at the DB level.
        with self.ManagedSessionMaker(read_only=False) as session:
            role_id_subq = (
                session.query(SqlRole.id).filter(SqlRole.workspace == workspace_name).subquery()
            )
            session.query(SqlRolePermission).filter(
                SqlRolePermission.role_id.in_(select(role_id_subq))
            ).delete(synchronize_session=False)
            session.query(SqlUserRoleAssignment).filter(
                SqlUserRoleAssignment.role_id.in_(select(role_id_subq))
            ).delete(synchronize_session=False)
            session.query(SqlRole).filter(SqlRole.workspace == workspace_name).delete(
                synchronize_session=False
            )

    # ---- RolePermission CRUD ----

    def add_role_permission(
        self,
        role_id: int,
        resource_type: str,
        resource_pattern: str,
        permission: str,
    ) -> RolePermission:
        _validate_permission_for_resource_type(permission, resource_type)
        # A pattern the type's grain does not allow would be silently ignored by the
        # resolver, so reject it up front. This covers the workspace slot (wildcard only)
        # and every sub-resource type (also wildcard only, until search-filter push-down
        # can enforce a per-id child grant in list paths as well as point routes).
        _validate_resource_pattern(resource_pattern, resource_type)
        with self.ManagedSessionMaker(read_only=False) as session:
            self._get_role(session, role_id)
            try:
                rp = SqlRolePermission(
                    role_id=role_id,
                    resource_type=resource_type,
                    resource_pattern=resource_pattern,
                    permission=permission,
                )
                session.add(rp)
                session.flush()
                return rp.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Role permission (role_id={role_id}, resource_type={resource_type}, "
                    f"resource_pattern={resource_pattern}) already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                ) from e

    @staticmethod
    def _get_role_permission(session, role_permission_id: int) -> SqlRolePermission:
        try:
            return (
                session
                .query(SqlRolePermission)
                .filter(SqlRolePermission.id == role_permission_id)
                .one()
            )
        except NoResultFound:
            raise MlflowException(
                f"Role permission with id={role_permission_id} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                f"Found multiple role permissions with id={role_permission_id}",
                INVALID_STATE,
            )

    def get_role_permission(self, role_permission_id: int) -> RolePermission:
        with self.ManagedSessionMaker() as session:
            return self._get_role_permission(session, role_permission_id).to_mlflow_entity()

    def remove_role_permission(self, role_permission_id: int) -> None:
        with self.ManagedSessionMaker(read_only=False) as session:
            rp = self._get_role_permission(session, role_permission_id)
            session.delete(rp)

    def list_role_permissions(self, role_id: int) -> list[RolePermission]:
        with self.ManagedSessionMaker() as session:
            self._get_role(session, role_id)
            perms = (
                session.query(SqlRolePermission).filter(SqlRolePermission.role_id == role_id).all()
            )
            return [p.to_mlflow_entity() for p in perms]

    def update_role_permission(self, role_permission_id: int, permission: str) -> RolePermission:
        with self.ManagedSessionMaker(read_only=False) as session:
            rp = self._get_role_permission(session, role_permission_id)
            _validate_permission_for_resource_type(permission, rp.resource_type)
            rp.permission = permission
            return rp.to_mlflow_entity()

    # ---- MutationConditions CRUD ----
    #
    # Two optional filters per (role, resource_type) that gate create/mutation only.
    # Conditions subtract from what grants allow and never confer access, so an
    # empty table reproduces the pre-conditions behaviour exactly.

    def add_mutation_condition(
        self,
        role_id: int,
        resource_type: str,
        *,
        resource_pattern: "str | None" = None,
        container_resource_type: "str | None" = None,
        container_resource_pattern: "str | None" = None,
        value_condition: str | None = None,
        target_condition: str | None = None,
    ) -> MutationConditions:
        validate_condition_resource_type(resource_type)
        resource_pattern, container_resource_type, container_resource_pattern = (
            normalize_condition_scope(
                resource_type,
                resource_pattern,
                container_resource_type,
                container_resource_pattern,
            )
        )
        # Validate here, not at evaluation time. A condition that failed to parse
        # mid-request would have to either fail open (unsafe) or deny every mutation
        # (an outage), so the only good place to catch it is on the way in.
        validate_condition(value_condition, NAMESPACE_REQUEST, resource_type)
        validate_condition(target_condition, NAMESPACE_RESOURCE, resource_type)
        if value_condition is None and target_condition is None:
            raise MlflowException(
                "A mutation condition needs at least one of 'value_condition' or "
                "'target_condition'. An object with neither restricts nothing, so it would "
                "occupy a slot while reading as a configured restriction.",
                INVALID_PARAMETER_VALUE,
            )
        with self.ManagedSessionMaker(read_only=False) as session:
            self._get_role(session, role_id)
            return self._insert_mutation_condition(
                session,
                role_id,
                resource_type,
                resource_pattern=resource_pattern,
                container_resource_type=container_resource_type,
                container_resource_pattern=container_resource_pattern,
                value_condition=value_condition,
                target_condition=target_condition,
            )

    def add_user_mutation_condition(
        self,
        username: str,
        resource_type: str,
        *,
        resource_pattern: "str | None" = None,
        container_resource_type: "str | None" = None,
        container_resource_pattern: "str | None" = None,
        value_condition: str | None = None,
        target_condition: str | None = None,
    ) -> MutationConditions:
        """Add a condition to ``username``'s synthetic role in the active workspace.

        The user-addressed counterpart of ``grant_user_resource_permission``, and it
        exists for the same reason: a direct grant and a direct condition both belong to
        the hidden ``__user_<id>__`` role, and a caller naming a user should not have to
        know that, nor have to create the role first.

        Creating the role on demand is what makes the two symmetric. Requiring a direct
        grant before a direct condition could be added would be an ordering constraint
        with no model behind it -- the role is an implementation detail of how per-user
        access is stored, not something the caller asked for.

        Validation is shared with the role-addressed path, so an unparseable filter or a
        filterless object is refused here too, before any role is created.
        """
        validate_condition_resource_type(resource_type)
        resource_pattern, container_resource_type, container_resource_pattern = (
            normalize_condition_scope(
                resource_type,
                resource_pattern,
                container_resource_type,
                container_resource_pattern,
            )
        )
        validate_condition(value_condition, NAMESPACE_REQUEST, resource_type)
        validate_condition(target_condition, NAMESPACE_RESOURCE, resource_type)
        if value_condition is None and target_condition is None:
            raise MlflowException(
                "A mutation condition needs at least one of 'value_condition' or "
                "'target_condition'. An object with neither restricts nothing, so it would "
                "occupy a slot while reading as a configured restriction.",
                INVALID_PARAMETER_VALUE,
            )
        with self.ManagedSessionMaker(read_only=False) as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            role = self._get_or_create_synthetic_user_role(session, user.id, workspace_name)
            return self._insert_mutation_condition(
                session,
                role.id,
                resource_type,
                resource_pattern=resource_pattern,
                container_resource_type=container_resource_type,
                container_resource_pattern=container_resource_pattern,
                value_condition=value_condition,
                target_condition=target_condition,
            )

    def _insert_mutation_condition(
        self,
        session,
        role_id: int,
        resource_type: str,
        *,
        resource_pattern: str,
        container_resource_type: str,
        container_resource_pattern: str,
        value_condition: str | None,
        target_condition: str | None,
    ) -> MutationConditions:
        """Allocate a slot and insert, retrying on the UNIQUE.

        The slot is picked from what this transaction can see, so two concurrent adds can
        choose the same free slot; the constraint is what actually enforces the bound, and
        the loser simply picks again. A savepoint keeps the failed insert from poisoning
        the surrounding session.

        Takes a session rather than opening one so the user-addressed path can create the
        synthetic role and insert the condition in a single transaction -- otherwise a
        failure here would leave behind a role the caller never asked for.
        """
        for _ in range(_SLOT_ALLOCATION_ATTEMPTS):
            slot = self._allocate_condition_slot(session, role_id, resource_type)
            mc = SqlMutationConditions(
                role_id=role_id,
                resource_type=resource_type,
                condition_slot=slot,
                resource_pattern=resource_pattern,
                container_resource_type=container_resource_type,
                container_resource_pattern=container_resource_pattern,
                value_condition=value_condition,
                target_condition=target_condition,
            )
            try:
                with session.begin_nested():
                    session.add(mc)
                    session.flush()
            except IntegrityError:
                continue
            return mc.to_mlflow_entity()
        raise MlflowException(
            f"Could not allocate a condition slot for (role_id={role_id}, "
            f"resource_type={resource_type}) after {_SLOT_ALLOCATION_ATTEMPTS} attempts "
            f"because concurrent writers kept taking the chosen slot. Retry the request.",
            TEMPORARILY_UNAVAILABLE,
        )

    @staticmethod
    def _allocate_condition_slot(session, role_id: int, resource_type: str) -> int:
        """The lowest free slot for this ``(role, resource_type)``.

        Lowest-free rather than max-plus-one so that slots freed by ``remove`` are
        reused -- otherwise a role repeatedly adding and removing conditions would
        exhaust the range while holding almost none.

        Slots carry no ordering meaning: every applicable condition must pass, so
        there is nothing to order. The number exists only to make the bound
        enforceable by a constraint.
        """
        used = {
            slot
            for (slot,) in session
            .query(SqlMutationConditions.condition_slot)
            .filter(
                SqlMutationConditions.role_id == role_id,
                SqlMutationConditions.resource_type == resource_type,
            )
            .all()
        }
        for slot in range(1, MAX_CONDITIONS_PER_ROLE_TYPE + 1):
            if slot not in used:
                return slot
        raise MlflowException(
            f"Role {role_id} already has the maximum of {MAX_CONDITIONS_PER_ROLE_TYPE} "
            f"mutation conditions for resource type '{resource_type}'. Remove one before "
            f"adding another, or express the restriction in fewer objects -- conditions all "
            f"AND, so several narrow objects are often one broader one.",
            RESOURCE_ALREADY_EXISTS,
        )

    @staticmethod
    def _get_mutation_condition(session, condition_id: int):
        try:
            return (
                session
                .query(SqlMutationConditions)
                .filter(SqlMutationConditions.id == condition_id)
                .one()
            )
        except NoResultFound:
            raise MlflowException(
                f"Mutation condition with id={condition_id} not found",
                RESOURCE_DOES_NOT_EXIST,
            )

    def get_mutation_condition(self, condition_id: int) -> MutationConditions:
        with self.ManagedSessionMaker() as session:
            return self._get_mutation_condition(session, condition_id).to_mlflow_entity()

    def update_mutation_condition(
        self,
        condition_id: int,
        *,
        value_condition: str | None = None,
        target_condition: str | None = None,
        update_value_condition: bool = True,
        update_target_condition: bool = True,
        resource_pattern: "str | None" = None,
        container_resource_type: "str | None" = None,
        container_resource_pattern: "str | None" = None,
        update_scope: bool = False,
    ) -> "MutationConditions | None":
        """Partial update, addressed by condition id.

        Three distinct intents, which is why the ``update_*`` flags exist rather than
        overloading ``None``:

        - **set**: pass the new string with its flag true
        - **clear**: pass ``None`` with its flag true (removes that restriction)
        - **leave unchanged**: pass its flag false

        Without the flags, "clear the target condition" and "leave the target
        condition alone" would both arrive as ``None`` -- and guessing wrong in the
        clearing direction silently removes a restriction an admin still wants.

        The parent scope is replaced as a **pair** under its own flag: an exact
        ``(type, id)`` or two ``None``s for unscoped. Updating one half alone is not
        expressible, because half a scope is not a scope.

        Clearing **both** filters deletes the object and returns ``None``. An object
        with neither filter restricts nothing, so keeping it would hold a slot while
        reading as a configured restriction -- the same reason ``add`` refuses one.
        """
        with self.ManagedSessionMaker(read_only=False) as session:
            mc = self._get_mutation_condition(session, condition_id)
            resource_type = mc.resource_type
            if update_value_condition:
                validate_condition(value_condition, NAMESPACE_REQUEST, resource_type)
                mc.value_condition = value_condition
            if update_target_condition:
                validate_condition(target_condition, NAMESPACE_RESOURCE, resource_type)
                mc.target_condition = target_condition
            if update_scope:
                # One flag for both axes, because they are validated together: whether a
                # container is legal depends on the resource type, and whether a wildcard
                # container collapses to the workspace depends on the container. Replacing
                # half a scope would skip that joint check.
                (
                    new_pattern,
                    new_container_type,
                    new_container_pattern,
                ) = normalize_condition_scope(
                    resource_type,
                    resource_pattern,
                    container_resource_type,
                    container_resource_pattern,
                )
                mc.resource_pattern = new_pattern
                mc.container_resource_type = new_container_type
                mc.container_resource_pattern = new_container_pattern
            if mc.value_condition is None and mc.target_condition is None:
                session.delete(mc)
                return None
            return mc.to_mlflow_entity()

    def remove_mutation_condition(self, condition_id: int) -> None:
        with self.ManagedSessionMaker(read_only=False) as session:
            mc = self._get_mutation_condition(session, condition_id)
            session.delete(mc)

    def list_mutation_conditions(self, role_id: int) -> list[MutationConditions]:
        with self.ManagedSessionMaker() as session:
            self._get_role(session, role_id)
            rows = (
                session
                .query(SqlMutationConditions)
                .filter(SqlMutationConditions.role_id == role_id)
                .order_by(
                    SqlMutationConditions.resource_type,
                    SqlMutationConditions.condition_slot,
                )
                .all()
            )
            return [r.to_mlflow_entity() for r in rows]

    def list_mutation_conditions_for_user(
        self,
        user_id: int,
        workspace: str,
        resource_types: "Collection[str]",
        parents: "Mapping[str, Collection[str]] | None" = None,
    ) -> list["MutationConditionRow"]:
        """Every condition applying to ``user_id`` in ``workspace`` for these types.

        One query, the same join shape as ``list_grants``. Two differences from the
        grants path, both because conditions only subtract:

        - **No workspace-wide fallback.** ``list_grants`` folds in
          ``resource_type='workspace'`` rows because a workspace grant applies to
          every type. A condition is not a grant, and a workspace-wide *restriction*
          is not something the RFC defines, so only the named types are queried.
        - **No precedence.** Rows come back flat and the caller ANDs them. There is
          nothing to fold, because two restrictions cannot conflict -- if either says
          no, the answer is no.

        Per-user conditions need no special-casing: a per-user grant lives on a
        synthetic ``__user_<id>__`` role that is a real ``roles`` row, so this join
        picks it up like any other role the user holds (D10).

        ``parents`` carries the resolved direct parent IDs in play for each type, and
        is what keeps the per-role-and-type bound a *storage* bound. The scope
        predicate runs in SQL, so a role holding the full
        ``MAX_CONDITIONS_PER_ROLE_TYPE`` for one type transfers only the conditions
        that actually apply to this request -- the unscoped ones plus those naming a
        parent in play. Filtering in Python instead would transfer all of them on
        every request and turn the bound into a per-request cost.

        A type absent from ``parents``, or mapped to no IDs, matches only unscoped
        conditions. That is the correct reading and not a fail-open: a scoped
        condition governs children of a named parent, so a request whose parent could
        not be resolved is outside every scope. The *refusal* for an unresolvable
        parent belongs at the context-construction boundary, where the caller knows it
        had a child to govern -- by the time a query runs, an empty parent set and a
        genuinely parentless target are indistinguishable.
        """
        types = set(resource_types)
        for resource_type in types:
            validate_condition_resource_type(resource_type)
        if not types:
            return []
        parents = parents or {}
        with self.ManagedSessionMaker() as session:
            rows = (
                session
                .query(
                    SqlMutationConditions.resource_type,
                    SqlMutationConditions.value_condition,
                    SqlMutationConditions.target_condition,
                    SqlMutationConditions.resource_pattern,
                )
                .join(SqlRole, SqlRole.id == SqlMutationConditions.role_id)
                .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
                .filter(
                    SqlUserRoleAssignment.user_id == user_id,
                    SqlRole.workspace == workspace,
                    or_(*self._scope_predicates(types, parents)),
                )
                .all()
            )
            return [
                MutationConditionRow(rtype, value, target, pattern)
                for rtype, value, target, pattern in rows
            ]

    @staticmethod
    def _scope_predicates(types: "Collection[str]", parents: "Mapping[str, Collection[str]]"):
        """One predicate per type in play: workspace-wide, or a container in play.

        Every condition has a container, so this is an equality test rather than a
        null check: ``workspace`` means no containment narrowing and matches always,
        and any other container must name an id the request actually resolved.

        Built per type rather than as a single ``resource_type IN (...)`` plus one shared
        container filter, because the containers in play differ by type. Under a shared
        filter a request touching runs of experiment 7 would also pull in a version
        condition contained by registered model 7 -- the id would satisfy a container
        belonging to another type. Each type therefore gets its own container set.

        The ``container_resource_type`` equality is defence in depth rather than what
        provides that isolation: the per-type ``resource_type ==`` already partitions the
        rows, and :func:`normalize_condition_scope` refuses to store a row whose container
        is not its resource type's declared one. It costs nothing and keeps a corrupt row
        from matching.

        A wildcard container never appears here because it is normalised to ``workspace``
        on write, which is what keeps this to two branches.
        """
        predicates = []
        for resource_type in sorted(types):
            in_play = parents.get(resource_type) or ()
            unscoped = SqlMutationConditions.container_resource_type == CONTAINER_WORKSPACE
            if in_play:
                scope = or_(
                    unscoped,
                    and_(
                        SqlMutationConditions.container_resource_type
                        == PARENT_RESOURCE_TYPES.get(resource_type),
                        SqlMutationConditions.container_resource_pattern.in_(sorted(set(in_play))),
                    ),
                )
            else:
                scope = unscoped
            predicates.append(and_(SqlMutationConditions.resource_type == resource_type, scope))
        return predicates

    # ---- UserRoleAssignment CRUD ----

    def assign_role_to_user(self, user_id: int, role_id: int) -> UserRoleAssignment:
        with self.ManagedSessionMaker(read_only=False) as session:
            # Validate both user and role exist before attempting assignment
            user = session.get(SqlUser, user_id)
            if user is None:
                raise MlflowException(
                    f"User with id={user_id} not found",
                    RESOURCE_DOES_NOT_EXIST,
                )
            self._get_role(session, role_id)
            # Check for duplicate assignment before insert
            existing = (
                session
                .query(SqlUserRoleAssignment)
                .filter(
                    SqlUserRoleAssignment.user_id == user_id,
                    SqlUserRoleAssignment.role_id == role_id,
                )
                .first()
            )
            if existing is not None:
                raise MlflowException(
                    f"User role assignment (user_id={user_id}, role_id={role_id}) already exists",
                    RESOURCE_ALREADY_EXISTS,
                )
            assignment = SqlUserRoleAssignment(user_id=user_id, role_id=role_id)
            session.add(assignment)
            session.flush()
            return assignment.to_mlflow_entity()

    def unassign_role_from_user(self, user_id: int, role_id: int) -> None:
        with self.ManagedSessionMaker(read_only=False) as session:
            try:
                assignment = (
                    session
                    .query(SqlUserRoleAssignment)
                    .filter(
                        SqlUserRoleAssignment.user_id == user_id,
                        SqlUserRoleAssignment.role_id == role_id,
                    )
                    .one()
                )
            except NoResultFound:
                raise MlflowException(
                    f"User role assignment (user_id={user_id}, role_id={role_id}) not found",
                    RESOURCE_DOES_NOT_EXIST,
                )
            session.delete(assignment)

    def list_user_roles(self, user_id: int) -> list[Role]:
        with self.ManagedSessionMaker() as session:
            roles = (
                session
                .query(SqlRole)
                .options(selectinload(SqlRole.permissions))
                .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
                .filter(SqlUserRoleAssignment.user_id == user_id)
                .all()
            )
            return [r.to_mlflow_entity() for r in roles]

    def list_user_roles_for_workspace(self, user_id: int, workspace: str) -> list[Role]:
        with self.ManagedSessionMaker() as session:
            roles = (
                session
                .query(SqlRole)
                .options(selectinload(SqlRole.permissions))
                .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
                .filter(
                    SqlUserRoleAssignment.user_id == user_id,
                    SqlRole.workspace == workspace,
                )
                .all()
            )
            return [r.to_mlflow_entity() for r in roles]

    def user_has_any_role_in_workspace(self, user_id: int, workspace: str) -> bool:
        """
        Lightweight existence check — returns True iff the user has at least one role
        assignment in the given workspace. Used by validators that only need membership,
        not the full role entities.
        """
        with self.ManagedSessionMaker() as session:
            return (
                session
                .query(SqlUserRoleAssignment.id)
                .join(SqlRole, SqlRole.id == SqlUserRoleAssignment.role_id)
                .filter(
                    SqlUserRoleAssignment.user_id == user_id,
                    SqlRole.workspace == workspace,
                )
                .first()
                is not None
            )

    def list_user_present_workspaces(self, user_id: int) -> set[str]:
        """
        Return every workspace where the user has at least one role assignment.
        Bulk membership check used by validators that need to authorize a request
        spanning multiple workspaces in a single query.
        """
        with self.ManagedSessionMaker() as session:
            return self._user_present_workspaces(session, user_id)

    def list_role_users(self, role_id: int) -> list[UserRoleAssignment]:
        with self.ManagedSessionMaker() as session:
            self._get_role(session, role_id)
            assignments = (
                session
                .query(SqlUserRoleAssignment)
                .filter(SqlUserRoleAssignment.role_id == role_id)
                .all()
            )
            return [a.to_mlflow_entity() for a in assignments]

    # ---- Role-based permission resolution ----

    def get_role_permission_for_resource(
        self, user_id: int, resource_type: str, resource_id: str, workspace: str
    ) -> Permission | None:
        with self.ManagedSessionMaker() as session:
            roles = (
                session
                .query(SqlRole)
                .options(selectinload(SqlRole.permissions))
                .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
                .filter(
                    SqlUserRoleAssignment.user_id == user_id,
                    SqlRole.workspace == workspace,
                )
                .all()
            )
            if not roles:
                return None

            best_permission_name: str | None = None
            denied = False
            workspace_admin = False
            for role in roles:
                for rp in role.permissions:
                    # (workspace, *) folds into resource-type queries only for
                    # MANAGE (workspace admin); USE is the "member can join +
                    # create" signal and folds only for workspace-tier queries.
                    if rp.resource_type == RESOURCE_TYPE_WORKSPACE and rp.resource_pattern == "*":
                        if resource_type == RESOURCE_TYPE_WORKSPACE or rp.permission == MANAGE.name:
                            workspace_admin = workspace_admin or rp.permission == MANAGE.name
                            best_permission_name = (
                                max_permission(best_permission_name, rp.permission)
                                if best_permission_name is not None
                                else rp.permission
                            )
                        continue
                    # Resource-type-specific permission.
                    if rp.resource_type != resource_type:
                        continue
                    if rp.resource_pattern in ("*", resource_id):
                        # DENY is below every positive level, so folding it with
                        # ``max_permission`` would silently lift it to the positive grant
                        # beside it. Track it separately, as ``fold_grants_for_key`` does.
                        if rp.permission == DENY.name:
                            denied = True
                            continue
                        best_permission_name = (
                            max_permission(best_permission_name, rp.permission)
                            if best_permission_name is not None
                            else rp.permission
                        )

            # A workspace admin is not restrictable, so that precedes DENY -- the same
            # ordering ``resolve_permissions`` applies.
            if denied and not workspace_admin:
                return DENY
            if best_permission_name is None:
                return None
            return get_permission(best_permission_name)

    @staticmethod
    def _workspace_admin_workspaces(session, user_id: int) -> set[str]:
        """
        Return the set of workspaces where ``user_id`` is a workspace admin.

        A user is a workspace admin in ``workspace`` if they have any role in
        ``workspace`` carrying a
        ``(resource_type='workspace', resource_pattern='*', permission=MANAGE)``
        grant — the simplified model's single admin shape, used by both the
        seeded ``admin`` role and migrated legacy ``workspace_permissions(MANAGE)``
        rows.
        """
        rows = (
            session
            .query(SqlRole.workspace)
            .join(SqlRolePermission, SqlRole.id == SqlRolePermission.role_id)
            .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
            .filter(
                SqlUserRoleAssignment.user_id == user_id,
                SqlRolePermission.resource_type == RESOURCE_TYPE_WORKSPACE,
                SqlRolePermission.resource_pattern == "*",
                SqlRolePermission.permission == MANAGE.name,
            )
            .distinct()
            .all()
        )
        return {w for (w,) in rows}

    @staticmethod
    def _user_present_workspaces(session, user_id: int) -> set[str]:
        """
        Return every workspace the user has some presence in via a role assignment.
        Used when scoping cross-user authorization decisions.
        """
        rows = (
            session
            .query(SqlRole.workspace)
            .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
            .filter(SqlUserRoleAssignment.user_id == user_id)
            .distinct()
            .all()
        )
        return {w for (w,) in rows}

    def is_workspace_admin(self, user_id: int, workspace: str) -> bool:
        """
        True if the user is a workspace admin in ``workspace`` — i.e. they hold a role
        in that workspace with ``resource_pattern='*'``, ``permission=MANAGE``, and
        ``resource_type`` either ``'workspace'`` (explicit admin grant) or ``'*'`` (the
        workspace-wide MANAGE form the migration produces from legacy
        ``workspace_permissions`` rows).
        """
        with self.ManagedSessionMaker() as session:
            return workspace in self._workspace_admin_workspaces(session, user_id)

    def list_grants(
        self, user_id: int, workspace: str, resource_types: "Collection[str]"
    ) -> list["RoleGrantRow"]:
        """
        The user's role-based grants in ``workspace`` for ``resource_types``, plus the
        workspace-wide grants (which can apply to any type).
        """
        types = set(resource_types)
        for resource_type in types:
            _validate_resource_type(resource_type)
        types.add(RESOURCE_TYPE_WORKSPACE)
        with self.ManagedSessionMaker() as session:
            rows = (
                session
                .query(
                    SqlRolePermission.resource_type,
                    SqlRolePermission.resource_pattern,
                    SqlRolePermission.permission,
                )
                .join(SqlRole, SqlRole.id == SqlRolePermission.role_id)
                .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
                .filter(
                    SqlUserRoleAssignment.user_id == user_id,
                    SqlRole.workspace == workspace,
                    SqlRolePermission.resource_type.in_(types),
                )
                .all()
            )
            return [RoleGrantRow(rtype, pattern, permission) for rtype, pattern, permission in rows]

    def list_role_grants_for_user_in_workspace(
        self, user_id: int, workspace: str, resource_type: str
    ) -> list[tuple[str, str]]:
        """
        Return the user's **role-based** permission grants in ``workspace`` that apply
        to resources of ``resource_type``. Direct per-resource grants (e.g. rows in
        ``experiment_permissions``) are intentionally **not** included — callers that
        need the full authorization picture fold them in separately (see
        ``filter_experiment_ids``, which unions the result of this query with
        ``list_experiment_permissions`` from the legacy table).

        Includes both grants on the specific resource_type and workspace-wide grants
        (``resource_type='workspace'``, ``resource_pattern='*'``) since those apply to
        every resource type.

        Returns a list of ``(resource_pattern, permission)`` tuples.
        """
        _validate_resource_type(resource_type)
        with self.ManagedSessionMaker() as session:
            rows = (
                session
                .query(SqlRolePermission.resource_pattern, SqlRolePermission.permission)
                .join(SqlRole, SqlRole.id == SqlRolePermission.role_id)
                .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
                .filter(
                    SqlUserRoleAssignment.user_id == user_id,
                    SqlRole.workspace == workspace,
                    or_(
                        SqlRolePermission.resource_type == resource_type,
                        and_(
                            SqlRolePermission.resource_type == RESOURCE_TYPE_WORKSPACE,
                            SqlRolePermission.resource_pattern == "*",
                        ),
                    ),
                )
                .all()
            )
            return [(pattern, permission) for pattern, permission in rows]

    def list_workspace_admin_workspaces(self, user_id: int) -> set[str]:
        """
        Return the set of workspaces where ``user_id`` is a workspace admin. Includes
        both role-based admin grants
        (``(resource_type='workspace', resource_pattern='*', permission=MANAGE)``) and
        legacy ``workspace_permissions`` MANAGE grants. See
        ``_workspace_admin_workspaces`` for the consolidation rationale.
        """
        with self.ManagedSessionMaker() as session:
            return self._workspace_admin_workspaces(session, user_id)

    def is_workspace_admin_of_any_of_users_workspaces(
        self, admin_user_id: int, target_user_id: int
    ) -> bool:
        """
        True if ``admin_user_id`` is a workspace admin in at least one workspace where
        ``target_user_id`` has presence. Both sides consult role assignments AND legacy
        ``workspace_permissions`` so operators mid-migration see consistent behavior.
        """
        with self.ManagedSessionMaker() as session:
            admin_workspaces = self._workspace_admin_workspaces(session, admin_user_id)
            if not admin_workspaces:
                return False
            target_workspaces = self._user_present_workspaces(session, target_user_id)
            return bool(admin_workspaces & target_workspaces)
