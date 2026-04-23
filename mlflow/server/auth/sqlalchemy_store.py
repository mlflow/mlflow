import re

from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError, MultipleResultsFound, NoResultFound
from sqlalchemy.orm import selectinload, sessionmaker
from werkzeug.security import check_password_hash, generate_password_hash

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_STATE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.server.auth.db import utils as dbutils
from mlflow.server.auth.db.models import (
    SqlExperimentPermission,
    SqlGatewayEndpointPermission,
    SqlGatewayModelDefinitionPermission,
    SqlGatewaySecretPermission,
    SqlRegisteredModelPermission,
    SqlRole,
    SqlRolePermission,
    SqlScorerPermission,
    SqlUser,
    SqlUserRoleAssignment,
    SqlWorkspacePermission,
)
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
from mlflow.server.auth.permissions import (
    MANAGE,
    Permission,
    _validate_permission,
    _validate_resource_type,
    get_permission,
    max_permission,
)
from mlflow.store.db.utils import _get_managed_session_maker, create_sqlalchemy_engine_with_retry
from mlflow.utils import workspace_context
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import _validate_password, _validate_username
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


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

    def init_db(self, db_uri):
        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.engine = create_sqlalchemy_engine_with_retry(db_uri)
        dbutils.migrate_if_needed(self.engine, "head")
        SessionMaker = sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = _get_managed_session_maker(SessionMaker, self.db_type)

    def authenticate_user(self, username: str, password: str) -> bool:
        with self.ManagedSessionMaker() as session:
            try:
                user = self._get_user(session, username)
                return check_password_hash(user.password_hash, password)
            except MlflowException:
                return False

    def create_user(self, username: str, password: str, is_admin: bool = False) -> User:
        _validate_username(username)
        _validate_password(password)
        pwhash = generate_password_hash(password)
        with self.ManagedSessionMaker() as session:
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

    def has_user(self, username: str) -> bool:
        with self.ManagedSessionMaker() as session:
            return session.query(SqlUser).filter(SqlUser.username == username).first() is not None

    def get_user(self, username: str) -> User:
        with self.ManagedSessionMaker() as session:
            return self._get_user(session, username).to_mlflow_entity()

    def list_users(self) -> list[User]:
        with self.ManagedSessionMaker() as session:
            users = session.query(SqlUser).all()
            return [u.to_mlflow_entity() for u in users]

    def update_user(
        self, username: str, password: str | None = None, is_admin: bool | None = None
    ) -> User:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username)
            if password is not None:
                pwhash = generate_password_hash(password)
                user.password_hash = pwhash
            if is_admin is not None:
                user.is_admin = is_admin
            return user.to_mlflow_entity()

    def delete_user(self, username: str):
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username)
            # Clean up the user's synthetic per-user roles (see the synthetic role helpers
            # section below) so their role_permissions / user_role_assignments rows don't
            # leak and don't hit FK errors on backends that enforce referential integrity.
            synthetic_name = self._synthetic_user_role_name(user.id)
            synthetic_roles = session.query(SqlRole).filter(SqlRole.name == synthetic_name).all()
            for role in synthetic_roles:
                session.delete(role)
            session.flush()
            session.delete(user)

    # ---- Synthetic user-role helpers (dual-write into role_permissions) ----
    #
    # Every legacy per-resource grant (experiment_permissions, registered_model_permissions,
    # ...) is mirrored into `role_permissions` under a synthetic role named
    # `__user_<user_id>__` scoped to the grant's workspace. This keeps
    # `role_permissions` in sync with the per-resource tables so it can serve as an
    # authoritative permission source without a separate backfill.
    #
    # The synthetic namespace (``__user_<int>__``) is RESERVED: ``create_role`` and
    # ``update_role`` reject names matching the pattern so an API consumer can't hijack a
    # synthetic role to receive another user's mirrored grants. Bulk cleanup / rename
    # helpers (``_unmirror_resource``, ``_rename_mirrored_resource``) also restrict their
    # scope to synthetic roles so they never touch admin-created roles that happen to
    # hold overlapping grants.
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
        """Guard user-facing role CRUD against the reserved synthetic pattern."""
        if cls._is_synthetic_role_name(name):
            raise MlflowException.invalid_parameter_value(
                f"Role name {name!r} matches the reserved synthetic pattern "
                f"'__user_<id>__' used by the role_permissions dual-write. "
                "Choose a different name."
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
                role = (
                    session
                    .query(SqlRole)
                    .filter(SqlRole.workspace == workspace, SqlRole.name == name)
                    .one()
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

    def _mirror_user_grant(
        self,
        session,
        user_id: int,
        workspace: str,
        resource_type: str,
        resource_pattern: str,
        permission: str,
    ) -> None:
        role = self._get_or_create_synthetic_user_role(session, user_id, workspace)
        rp = (
            session
            .query(SqlRolePermission)
            .filter(
                SqlRolePermission.role_id == role.id,
                SqlRolePermission.resource_type == resource_type,
                SqlRolePermission.resource_pattern == resource_pattern,
            )
            .first()
        )
        if rp is None:
            session.add(
                SqlRolePermission(
                    role_id=role.id,
                    resource_type=resource_type,
                    resource_pattern=resource_pattern,
                    permission=permission,
                )
            )
        else:
            rp.permission = permission
        session.flush()

    def _unmirror_user_grant(
        self,
        session,
        user_id: int,
        workspace: str,
        resource_type: str,
        resource_pattern: str,
    ) -> None:
        name = self._synthetic_user_role_name(user_id)
        role = (
            session
            .query(SqlRole)
            .filter(SqlRole.workspace == workspace, SqlRole.name == name)
            .first()
        )
        if role is None:
            return
        (
            session
            .query(SqlRolePermission)
            .filter(
                SqlRolePermission.role_id == role.id,
                SqlRolePermission.resource_type == resource_type,
                SqlRolePermission.resource_pattern == resource_pattern,
            )
            .delete(synchronize_session=False)
        )
        session.flush()

    def _synthetic_role_ids(self, session, workspace: str | None = None) -> list[int]:
        """
        Return ids of synthetic ``__user_<id>__`` roles, optionally scoped to
        ``workspace``. Callers of bulk cleanup/rename helpers route through this so they
        never touch admin-created roles that happen to share a grant.
        """
        query = session.query(SqlRole.id, SqlRole.name)
        if workspace is not None:
            query = query.filter(SqlRole.workspace == workspace)
        return [rid for (rid, name) in query.all() if self._is_synthetic_role_name(name)]

    def _unmirror_resource(
        self,
        session,
        resource_type: str,
        resource_pattern: str,
        workspace: str | None = None,
    ) -> None:
        # Delete all mirrored role_permissions rows for one resource across users.
        # `workspace=None` is used when the underlying resource ID is globally unique
        # (scorer, gateway_*), so mirrored rows are removed regardless of which
        # synthetic role holds them. Restricted to synthetic roles so admin-created
        # roles with an overlapping grant are never touched.
        role_ids = self._synthetic_role_ids(session, workspace=workspace)
        if not role_ids:
            return
        session.query(SqlRolePermission).filter(
            SqlRolePermission.role_id.in_(role_ids),
            SqlRolePermission.resource_type == resource_type,
            SqlRolePermission.resource_pattern == resource_pattern,
        ).delete(synchronize_session=False)
        session.flush()

    def _rename_mirrored_resource(
        self,
        session,
        workspace: str,
        resource_type: str,
        old_pattern: str,
        new_pattern: str,
    ) -> None:
        # Synthetic-role-only, same rationale as _unmirror_resource above.
        role_ids = self._synthetic_role_ids(session, workspace=workspace)
        if not role_ids:
            return
        (
            session
            .query(SqlRolePermission)
            .filter(
                SqlRolePermission.role_id.in_(role_ids),
                SqlRolePermission.resource_type == resource_type,
                SqlRolePermission.resource_pattern == old_pattern,
            )
            .update({SqlRolePermission.resource_pattern: new_pattern}, synchronize_session=False)
        )
        session.flush()

    def create_experiment_permission(
        self, experiment_id: str, username: str, permission: str
    ) -> ExperimentPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            try:
                perm = SqlExperimentPermission(
                    experiment_id=experiment_id, user_id=user.id, permission=permission
                )
                session.add(perm)
                session.flush()
                entity = perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Experiment permission (experiment_id={experiment_id}, username={username}) "
                    f"already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                )
            self._mirror_user_grant(
                session, user.id, workspace_name, "experiment", experiment_id, permission
            )
            return entity

    def _get_experiment_permission(
        self, session, experiment_id: str, username: str
    ) -> SqlExperimentPermission:
        try:
            user = self._get_user(session, username=username)
            return (
                session
                .query(SqlExperimentPermission)
                .filter(
                    SqlExperimentPermission.experiment_id == experiment_id,
                    SqlExperimentPermission.user_id == user.id,
                )
                .one()
            )
        except NoResultFound:
            raise MlflowException(
                f"Experiment permission with experiment_id={experiment_id} and "
                f"username={username} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                f"Found multiple experiment permissions with experiment_id={experiment_id} "
                f"and username={username}",
                INVALID_STATE,
            )

    def get_experiment_permission(self, experiment_id: str, username: str) -> ExperimentPermission:
        with self.ManagedSessionMaker() as session:
            return self._get_experiment_permission(
                session, experiment_id, username
            ).to_mlflow_entity()

    def list_experiment_permissions(self, username: str) -> list[ExperimentPermission]:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            perms = (
                session
                .query(SqlExperimentPermission)
                .filter(SqlExperimentPermission.user_id == user.id)
                .all()
            )
            return [p.to_mlflow_entity() for p in perms]

    def update_experiment_permission(
        self, experiment_id: str, username: str, permission: str
    ) -> ExperimentPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            perm = self._get_experiment_permission(session, experiment_id, username)
            perm.permission = permission
            workspace_name = self._get_active_workspace_name()
            self._mirror_user_grant(
                session, perm.user_id, workspace_name, "experiment", experiment_id, permission
            )
            return perm.to_mlflow_entity()

    def delete_experiment_permission(self, experiment_id: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_experiment_permission(session, experiment_id, username)
            user_id = perm.user_id
            workspace_name = self._get_active_workspace_name()
            session.delete(perm)
            session.flush()
            self._unmirror_user_grant(session, user_id, workspace_name, "experiment", experiment_id)

    def delete_workspace_permissions_for_workspace(self, workspace_name: str) -> None:
        with self.ManagedSessionMaker() as session:
            session.query(SqlWorkspacePermission).filter(
                SqlWorkspacePermission.workspace == workspace_name
            ).delete(synchronize_session=False)
            self._unmirror_resource(session, "*", "*", workspace=workspace_name)

    def create_registered_model_permission(
        self, name: str, username: str, permission: str
    ) -> RegisteredModelPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            try:
                perm = SqlRegisteredModelPermission(
                    workspace=workspace_name,
                    name=name,
                    user_id=user.id,
                    permission=permission,
                )
                session.add(perm)
                session.flush()
                entity = perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    "Registered model permission "
                    f"with workspace={workspace_name}, name={name} and username={username} "
                    f"already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                )
            self._mirror_user_grant(
                session, user.id, workspace_name, "registered_model", name, permission
            )
            return entity

    def _get_registered_model_permission(
        self, session, name: str, username: str
    ) -> SqlRegisteredModelPermission:
        try:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            return (
                session
                .query(SqlRegisteredModelPermission)
                .filter(
                    SqlRegisteredModelPermission.workspace == workspace_name,
                    SqlRegisteredModelPermission.name == name,
                    SqlRegisteredModelPermission.user_id == user.id,
                )
                .one()
            )
        except NoResultFound:
            raise MlflowException(
                "Registered model permission "
                f"with workspace={workspace_name}, name={name} and username={username} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                "Registered model permission "
                f"with workspace={workspace_name}, name={name} and username={username} "
                "found multiple times",
                INVALID_STATE,
            )

    def get_registered_model_permission(
        self, name: str, username: str
    ) -> RegisteredModelPermission:
        with self.ManagedSessionMaker() as session:
            return self._get_registered_model_permission(session, name, username).to_mlflow_entity()

    def list_registered_model_permissions(self, username: str) -> list[RegisteredModelPermission]:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            perms = (
                session
                .query(SqlRegisteredModelPermission)
                .filter(
                    SqlRegisteredModelPermission.user_id == user.id,
                    SqlRegisteredModelPermission.workspace == workspace_name,
                )
                .all()
            )
            return [p.to_mlflow_entity() for p in perms]

    def update_registered_model_permission(
        self, name: str, username: str, permission: str
    ) -> RegisteredModelPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            perm = self._get_registered_model_permission(session, name, username)
            perm.permission = permission
            self._mirror_user_grant(
                session, perm.user_id, perm.workspace, "registered_model", name, permission
            )
            return perm.to_mlflow_entity()

    def delete_registered_model_permission(self, name: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_registered_model_permission(session, name, username)
            user_id = perm.user_id
            workspace_name = perm.workspace
            session.delete(perm)
            session.flush()
            self._unmirror_user_grant(session, user_id, workspace_name, "registered_model", name)

    def delete_registered_model_permissions(self, name: str) -> None:
        """
        Delete *all* registered model permission rows for the given model name in the active
        workspace.

        This is primarily used as cleanup when a registered model is deleted to ensure that
        previously granted permissions do not implicitly carry over if a new model is later created
        with the same name.
        """
        with self.ManagedSessionMaker() as session:
            workspace_name = self._get_active_workspace_name()
            session.query(SqlRegisteredModelPermission).filter(
                SqlRegisteredModelPermission.workspace == workspace_name,
                SqlRegisteredModelPermission.name == name,
            ).delete(synchronize_session=False)
            self._unmirror_resource(session, "registered_model", name, workspace=workspace_name)

    def rename_registered_model_permissions(self, old_name: str, new_name: str):
        with self.ManagedSessionMaker() as session:
            workspace_name = self._get_active_workspace_name()
            perms = (
                session
                .query(SqlRegisteredModelPermission)
                .filter(
                    SqlRegisteredModelPermission.workspace == workspace_name,
                    SqlRegisteredModelPermission.name == old_name,
                )
                .all()
            )
            for perm in perms:
                perm.name = new_name
            self._rename_mirrored_resource(
                session, workspace_name, "registered_model", old_name, new_name
            )

    def list_workspace_permissions(self, workspace_name: str) -> list[WorkspacePermission]:
        with self.ManagedSessionMaker() as session:
            rows = (
                session
                .query(SqlWorkspacePermission)
                .filter(SqlWorkspacePermission.workspace == workspace_name)
                .all()
            )
            return [row.to_mlflow_entity() for row in rows]

    def list_user_workspace_permissions(self, username: str) -> list[WorkspacePermission]:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            rows = (
                session
                .query(SqlWorkspacePermission)
                .filter(SqlWorkspacePermission.user_id == user.id)
                .all()
            )
            return [row.to_mlflow_entity() for row in rows]

    def set_workspace_permission(
        self, workspace_name: str, username: str, permission: str
    ) -> WorkspacePermission:
        _validate_permission(permission)

        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            entity = session.get(
                SqlWorkspacePermission,
                (workspace_name, user.id),
            )
            if entity is None:
                entity = SqlWorkspacePermission(
                    workspace=workspace_name,
                    user_id=user.id,
                    permission=permission,
                )
                session.add(entity)
            else:
                entity.permission = permission
            session.flush()
            self._mirror_user_grant(session, user.id, workspace_name, "*", "*", permission)
            return entity.to_mlflow_entity()

    def delete_workspace_permission(self, workspace_name: str, username: str) -> None:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            entity = session.get(
                SqlWorkspacePermission,
                (workspace_name, user.id),
            )
            if entity is None:
                raise MlflowException(
                    (
                        "Workspace permission does not exist for "
                        f"workspace='{workspace_name}', username='{username}'"
                    ),
                    RESOURCE_DOES_NOT_EXIST,
                )
            session.delete(entity)
            session.flush()
            self._unmirror_user_grant(session, user.id, workspace_name, "*", "*")

    def list_accessible_workspace_names(self, username: str | None) -> set[str]:
        if username is None:
            return set()

        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            rows: list[SqlWorkspacePermission] = (
                session
                .query(SqlWorkspacePermission)
                .filter(SqlWorkspacePermission.user_id == user.id)
                .all()
            )
            accessible: set[str] = set()
            for row in rows:
                permission = row.permission
                if get_permission(permission).can_read:
                    accessible.add(row.workspace)
            return accessible

    def get_workspace_permission(self, workspace_name: str, username: str) -> Permission | None:
        """
        Get the workspace permission for a user.
        """
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            entity = session.get(SqlWorkspacePermission, (workspace_name, user.id))
            if entity is not None:
                return get_permission(entity.permission)
        return None

    def get_workspace_permission_via_roles(
        self, username: str, workspace_name: str
    ) -> Permission | None:
        """
        Role-table analog of ``get_workspace_permission``. Returns the union of
        ``(resource_type='*', resource_pattern='*')`` and
        ``(resource_type='workspace', resource_pattern='*')`` grants the user holds in
        ``workspace_name``. Used by the unified-reads code path.
        """
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            rows = (
                session
                .query(SqlRolePermission.permission)
                .join(SqlRole, SqlRole.id == SqlRolePermission.role_id)
                .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
                .filter(
                    SqlUserRoleAssignment.user_id == user.id,
                    SqlRole.workspace == workspace_name,
                    SqlRolePermission.resource_type.in_(("*", "workspace")),
                    SqlRolePermission.resource_pattern == "*",
                )
                .all()
            )
        if not rows:
            return None
        best: str | None = None
        for (perm,) in rows:
            best = perm if best is None else max_permission(best, perm)
        return get_permission(best) if best is not None else None

    def list_accessible_workspace_names_via_roles(self, username: str | None) -> set[str]:
        """
        Role-table analog of ``list_accessible_workspace_names``. Returns every workspace
        in which the user has at least one ``can_read``-capable ``role_permissions`` row,
        regardless of resource_type. Used by the unified-reads code path.
        """
        if username is None:
            return set()
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            rows = (
                session
                .query(SqlRole.workspace, SqlRolePermission.permission)
                .join(SqlRolePermission, SqlRole.id == SqlRolePermission.role_id)
                .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
                .filter(SqlUserRoleAssignment.user_id == user.id)
                .all()
            )
        accessible: set[str] = set()
        for workspace, permission in rows:
            if get_permission(permission).can_read:
                accessible.add(workspace)
        return accessible

    def create_scorer_permission(
        self, experiment_id: str, scorer_name: str, username: str, permission: str
    ) -> ScorerPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            try:
                perm = SqlScorerPermission(
                    experiment_id=experiment_id,
                    scorer_name=scorer_name,
                    user_id=user.id,
                    permission=permission,
                )
                session.add(perm)
                session.flush()
                entity = perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Scorer permission (experiment_id={experiment_id}, scorer_name={scorer_name}, "
                    f"username={username}) already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                ) from e
            self._mirror_user_grant(
                session,
                user.id,
                workspace_name,
                "scorer",
                f"{experiment_id}/{scorer_name}",
                permission,
            )
            return entity

    def _get_scorer_permission(
        self, session, experiment_id: str, scorer_name: str, username: str
    ) -> SqlScorerPermission:
        try:
            user = self._get_user(session, username=username)
            return (
                session
                .query(SqlScorerPermission)
                .filter(
                    SqlScorerPermission.experiment_id == experiment_id,
                    SqlScorerPermission.scorer_name == scorer_name,
                    SqlScorerPermission.user_id == user.id,
                )
                .one()
            )
        except NoResultFound:
            raise MlflowException(
                f"Scorer permission with experiment_id={experiment_id}, "
                f"scorer_name={scorer_name}, and username={username} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                f"Found multiple scorer permissions with experiment_id={experiment_id}, "
                f"scorer_name={scorer_name}, and username={username}",
                INVALID_STATE,
            )

    def get_scorer_permission(
        self, experiment_id: str, scorer_name: str, username: str
    ) -> ScorerPermission:
        with self.ManagedSessionMaker() as session:
            return self._get_scorer_permission(
                session, experiment_id, scorer_name, username
            ).to_mlflow_entity()

    def list_scorer_permissions(self, username: str) -> list[ScorerPermission]:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            perms = (
                session
                .query(SqlScorerPermission)
                .filter(SqlScorerPermission.user_id == user.id)
                .all()
            )
            return [p.to_mlflow_entity() for p in perms]

    def update_scorer_permission(
        self, experiment_id: str, scorer_name: str, username: str, permission: str
    ) -> ScorerPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            perm = self._get_scorer_permission(session, experiment_id, scorer_name, username)
            perm.permission = permission
            workspace_name = self._get_active_workspace_name()
            self._mirror_user_grant(
                session,
                perm.user_id,
                workspace_name,
                "scorer",
                f"{experiment_id}/{scorer_name}",
                permission,
            )
            return perm.to_mlflow_entity()

    def delete_scorer_permission(self, experiment_id: str, scorer_name: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_scorer_permission(session, experiment_id, scorer_name, username)
            user_id = perm.user_id
            workspace_name = self._get_active_workspace_name()
            session.delete(perm)
            session.flush()
            self._unmirror_user_grant(
                session,
                user_id,
                workspace_name,
                "scorer",
                f"{experiment_id}/{scorer_name}",
            )

    def delete_scorer_permissions_for_scorer(self, experiment_id: str, scorer_name: str):
        with self.ManagedSessionMaker() as session:
            session.query(SqlScorerPermission).filter(
                SqlScorerPermission.experiment_id == experiment_id,
                SqlScorerPermission.scorer_name == scorer_name,
            ).delete()
            self._unmirror_resource(session, "scorer", f"{experiment_id}/{scorer_name}")

    def create_gateway_secret_permission(
        self, secret_id: str, username: str, permission: str
    ) -> GatewaySecretPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            try:
                perm = SqlGatewaySecretPermission(
                    secret_id=secret_id, user_id=user.id, permission=permission
                )
                session.add(perm)
                session.flush()
                entity = perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Gateway secret permission (secret_id={secret_id}, username={username}) "
                    f"already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                ) from e
            self._mirror_user_grant(
                session, user.id, workspace_name, "gateway_secret", secret_id, permission
            )
            return entity

    def _get_gateway_secret_permission(
        self, session, secret_id: str, username: str
    ) -> SqlGatewaySecretPermission:
        try:
            user = self._get_user(session, username=username)
            return (
                session
                .query(SqlGatewaySecretPermission)
                .filter(
                    SqlGatewaySecretPermission.secret_id == secret_id,
                    SqlGatewaySecretPermission.user_id == user.id,
                )
                .one()
            )
        except NoResultFound:
            raise MlflowException(
                f"Gateway secret permission with secret_id={secret_id} and "
                f"username={username} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                f"Found multiple gateway secret permissions with secret_id={secret_id} "
                f"and username={username}",
                INVALID_STATE,
            )

    def get_gateway_secret_permission(
        self, secret_id: str, username: str
    ) -> GatewaySecretPermission:
        with self.ManagedSessionMaker() as session:
            return self._get_gateway_secret_permission(
                session, secret_id, username
            ).to_mlflow_entity()

    def list_gateway_secret_permissions(self, username: str) -> list[GatewaySecretPermission]:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            perms = (
                session
                .query(SqlGatewaySecretPermission)
                .filter(SqlGatewaySecretPermission.user_id == user.id)
                .all()
            )
            return [p.to_mlflow_entity() for p in perms]

    def update_gateway_secret_permission(
        self, secret_id: str, username: str, permission: str
    ) -> GatewaySecretPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            perm = self._get_gateway_secret_permission(session, secret_id, username)
            perm.permission = permission
            workspace_name = self._get_active_workspace_name()
            self._mirror_user_grant(
                session, perm.user_id, workspace_name, "gateway_secret", secret_id, permission
            )
            return perm.to_mlflow_entity()

    def delete_gateway_secret_permission(self, secret_id: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_gateway_secret_permission(session, secret_id, username)
            user_id = perm.user_id
            workspace_name = self._get_active_workspace_name()
            session.delete(perm)
            session.flush()
            self._unmirror_user_grant(session, user_id, workspace_name, "gateway_secret", secret_id)

    def delete_gateway_secret_permissions_for_secret(self, secret_id: str):
        with self.ManagedSessionMaker() as session:
            session.query(SqlGatewaySecretPermission).filter(
                SqlGatewaySecretPermission.secret_id == secret_id,
            ).delete()
            self._unmirror_resource(session, "gateway_secret", secret_id)

    def create_gateway_endpoint_permission(
        self, endpoint_id: str, username: str, permission: str
    ) -> GatewayEndpointPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            try:
                perm = SqlGatewayEndpointPermission(
                    endpoint_id=endpoint_id, user_id=user.id, permission=permission
                )
                session.add(perm)
                session.flush()
                entity = perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Gateway endpoint permission (endpoint_id={endpoint_id}, username={username}) "
                    f"already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                ) from e
            self._mirror_user_grant(
                session, user.id, workspace_name, "gateway_endpoint", endpoint_id, permission
            )
            return entity

    def _get_gateway_endpoint_permission(
        self, session, endpoint_id: str, username: str
    ) -> SqlGatewayEndpointPermission:
        try:
            user = self._get_user(session, username=username)
            return (
                session
                .query(SqlGatewayEndpointPermission)
                .filter(
                    SqlGatewayEndpointPermission.endpoint_id == endpoint_id,
                    SqlGatewayEndpointPermission.user_id == user.id,
                )
                .one()
            )
        except NoResultFound:
            raise MlflowException(
                f"Gateway endpoint permission with endpoint_id={endpoint_id} and "
                f"username={username} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                f"Found multiple gateway endpoint permissions with endpoint_id={endpoint_id} "
                f"and username={username}",
                INVALID_STATE,
            )

    def get_gateway_endpoint_permission(
        self, endpoint_id: str, username: str
    ) -> GatewayEndpointPermission:
        with self.ManagedSessionMaker() as session:
            return self._get_gateway_endpoint_permission(
                session, endpoint_id, username
            ).to_mlflow_entity()

    def list_gateway_endpoint_permissions(self, username: str) -> list[GatewayEndpointPermission]:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            perms = (
                session
                .query(SqlGatewayEndpointPermission)
                .filter(SqlGatewayEndpointPermission.user_id == user.id)
                .all()
            )
            return [p.to_mlflow_entity() for p in perms]

    def update_gateway_endpoint_permission(
        self, endpoint_id: str, username: str, permission: str
    ) -> GatewayEndpointPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            perm = self._get_gateway_endpoint_permission(session, endpoint_id, username)
            perm.permission = permission
            workspace_name = self._get_active_workspace_name()
            self._mirror_user_grant(
                session, perm.user_id, workspace_name, "gateway_endpoint", endpoint_id, permission
            )
            return perm.to_mlflow_entity()

    def delete_gateway_endpoint_permission(self, endpoint_id: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_gateway_endpoint_permission(session, endpoint_id, username)
            user_id = perm.user_id
            workspace_name = self._get_active_workspace_name()
            session.delete(perm)
            session.flush()
            self._unmirror_user_grant(
                session, user_id, workspace_name, "gateway_endpoint", endpoint_id
            )

    def delete_gateway_endpoint_permissions_for_endpoint(self, endpoint_id: str):
        with self.ManagedSessionMaker() as session:
            session.query(SqlGatewayEndpointPermission).filter(
                SqlGatewayEndpointPermission.endpoint_id == endpoint_id,
            ).delete()
            self._unmirror_resource(session, "gateway_endpoint", endpoint_id)

    def create_gateway_model_definition_permission(
        self, model_definition_id: str, username: str, permission: str
    ) -> GatewayModelDefinitionPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            workspace_name = self._get_active_workspace_name()
            try:
                perm = SqlGatewayModelDefinitionPermission(
                    model_definition_id=model_definition_id, user_id=user.id, permission=permission
                )
                session.add(perm)
                session.flush()
                entity = perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Gateway model definition permission "
                    f"(model_definition_id={model_definition_id}, username={username}) "
                    f"already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                ) from e
            self._mirror_user_grant(
                session,
                user.id,
                workspace_name,
                "gateway_model_definition",
                model_definition_id,
                permission,
            )
            return entity

    def _get_gateway_model_definition_permission(
        self, session, model_definition_id: str, username: str
    ) -> SqlGatewayModelDefinitionPermission:
        try:
            user = self._get_user(session, username=username)
            return (
                session
                .query(SqlGatewayModelDefinitionPermission)
                .filter(
                    SqlGatewayModelDefinitionPermission.model_definition_id == model_definition_id,
                    SqlGatewayModelDefinitionPermission.user_id == user.id,
                )
                .one()
            )
        except NoResultFound:
            raise MlflowException(
                f"Gateway model definition permission with "
                f"model_definition_id={model_definition_id} and username={username} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                f"Found multiple gateway model definition permissions with "
                f"model_definition_id={model_definition_id} and username={username}",
                INVALID_STATE,
            )

    def get_gateway_model_definition_permission(
        self, model_definition_id: str, username: str
    ) -> GatewayModelDefinitionPermission:
        with self.ManagedSessionMaker() as session:
            return self._get_gateway_model_definition_permission(
                session, model_definition_id, username
            ).to_mlflow_entity()

    def list_gateway_model_definition_permissions(
        self, username: str
    ) -> list[GatewayModelDefinitionPermission]:
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            perms = (
                session
                .query(SqlGatewayModelDefinitionPermission)
                .filter(SqlGatewayModelDefinitionPermission.user_id == user.id)
                .all()
            )
            return [p.to_mlflow_entity() for p in perms]

    def update_gateway_model_definition_permission(
        self, model_definition_id: str, username: str, permission: str
    ) -> GatewayModelDefinitionPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            perm = self._get_gateway_model_definition_permission(
                session, model_definition_id, username
            )
            perm.permission = permission
            workspace_name = self._get_active_workspace_name()
            self._mirror_user_grant(
                session,
                perm.user_id,
                workspace_name,
                "gateway_model_definition",
                model_definition_id,
                permission,
            )
            return perm.to_mlflow_entity()

    def delete_gateway_model_definition_permission(self, model_definition_id: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_gateway_model_definition_permission(
                session, model_definition_id, username
            )
            user_id = perm.user_id
            workspace_name = self._get_active_workspace_name()
            session.delete(perm)
            session.flush()
            self._unmirror_user_grant(
                session,
                user_id,
                workspace_name,
                "gateway_model_definition",
                model_definition_id,
            )

    def delete_gateway_model_definition_permissions_for_model_definition(
        self, model_definition_id: str
    ):
        with self.ManagedSessionMaker() as session:
            session.query(SqlGatewayModelDefinitionPermission).filter(
                SqlGatewayModelDefinitionPermission.model_definition_id == model_definition_id,
            ).delete()
            self._unmirror_resource(session, "gateway_model_definition", model_definition_id)

    # ---- Role CRUD ----

    def create_role(
        self,
        name: str,
        workspace: str,
        description: str | None = None,
    ) -> Role:
        self._reject_synthetic_role_name(name)
        with self.ManagedSessionMaker() as session:
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

    def list_roles(self, workspace: str) -> list[Role]:
        with self.ManagedSessionMaker() as session:
            roles = (
                session
                .query(SqlRole)
                .options(selectinload(SqlRole.permissions))
                .filter(SqlRole.workspace == workspace)
                .all()
            )
            return [r.to_mlflow_entity() for r in roles]

    def list_all_roles(self) -> list[Role]:
        with self.ManagedSessionMaker() as session:
            roles = session.query(SqlRole).options(selectinload(SqlRole.permissions)).all()
            return [r.to_mlflow_entity() for r in roles]

    def update_role(
        self,
        role_id: int,
        name: str | None = None,
        description: str | None = None,
    ) -> Role:
        if name is not None:
            self._reject_synthetic_role_name(name)
        with self.ManagedSessionMaker() as session:
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
        with self.ManagedSessionMaker() as session:
            role = self._get_role(session, role_id)
            session.delete(role)

    def delete_roles_for_workspace(self, workspace_name: str) -> None:
        # Batch delete: ORM-level ``cascade="all, delete-orphan"`` only fires when calling
        # ``session.delete(instance)``, so for a bulk delete we must explicitly remove
        # child rows (``role_permissions``, ``user_role_assignments``) before the roles
        # themselves. The FK doesn't declare ``ON DELETE CASCADE`` at the DB level.
        with self.ManagedSessionMaker() as session:
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
        _validate_permission(permission)
        _validate_resource_type(resource_type)
        # Workspace-scope and type-wildcard grants only support the "*" pattern. Any other
        # pattern would be silently ignored by the resolver, so reject it up front.
        if resource_type in ("workspace", "*") and resource_pattern != "*":
            raise MlflowException.invalid_parameter_value(
                f"resource_type='{resource_type}' requires resource_pattern='*'. "
                f"Got resource_pattern='{resource_pattern}'."
            )
        with self.ManagedSessionMaker() as session:
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
        with self.ManagedSessionMaker() as session:
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
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            rp = self._get_role_permission(session, role_permission_id)
            rp.permission = permission
            return rp.to_mlflow_entity()

    # ---- UserRoleAssignment CRUD ----

    def assign_role_to_user(self, user_id: int, role_id: int) -> UserRoleAssignment:
        with self.ManagedSessionMaker() as session:
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
        with self.ManagedSessionMaker() as session:
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
            for role in roles:
                for rp in role.permissions:
                    # Workspace-wide permission — applies to every resource type. Both
                    # ``resource_type='workspace'`` (workspace admin) and ``resource_type='*'``
                    # (type-wildcard, mirrors the legacy workspace_permissions table) match
                    # every resource in the role's workspace.
                    if rp.resource_type in ("workspace", "*") and rp.resource_pattern == "*":
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
                        best_permission_name = (
                            max_permission(best_permission_name, rp.permission)
                            if best_permission_name is not None
                            else rp.permission
                        )

            if best_permission_name is None:
                return None
            return get_permission(best_permission_name)

    @staticmethod
    def _workspace_admin_workspaces(session, user_id: int) -> set[str]:
        """
        Return the set of workspaces where ``user_id`` is a workspace admin, drawing
        from BOTH sources of truth:

        - Role-based: a role in the workspace with
          ``(resource_type='workspace', resource_pattern='*', permission=MANAGE)``.
        - Legacy: a ``workspace_permissions`` row with ``permission=MANAGE``. Pre-RBAC
          this was the only way to express workspace-wide admin authority; operators
          upgrading from pre-RBAC deployments retain that admin status until they
          migrate the grants into roles.
        """
        role_rows = (
            session
            .query(SqlRole.workspace)
            .join(SqlRolePermission, SqlRole.id == SqlRolePermission.role_id)
            .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
            .filter(
                SqlUserRoleAssignment.user_id == user_id,
                SqlRolePermission.resource_type == "workspace",
                SqlRolePermission.resource_pattern == "*",
                SqlRolePermission.permission == MANAGE.name,
            )
            .distinct()
            .all()
        )
        legacy_rows = (
            session
            .query(SqlWorkspacePermission.workspace)
            .filter(
                SqlWorkspacePermission.user_id == user_id,
                SqlWorkspacePermission.permission == MANAGE.name,
            )
            .distinct()
            .all()
        )
        return {w for (w,) in role_rows} | {w for (w,) in legacy_rows}

    @staticmethod
    def _user_present_workspaces(session, user_id: int) -> set[str]:
        """
        Return every workspace the user has some presence in — either via a role
        assignment or via a legacy ``workspace_permissions`` grant (of any level).
        Used when scoping cross-user authorization decisions.
        """
        role_rows = (
            session
            .query(SqlRole.workspace)
            .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
            .filter(SqlUserRoleAssignment.user_id == user_id)
            .distinct()
            .all()
        )
        legacy_rows = (
            session
            .query(SqlWorkspacePermission.workspace)
            .filter(SqlWorkspacePermission.user_id == user_id)
            .distinct()
            .all()
        )
        return {w for (w,) in role_rows} | {w for (w,) in legacy_rows}

    def is_workspace_admin(self, user_id: int, workspace: str) -> bool:
        """
        True if the user is a workspace admin in ``workspace``, via either a role
        (``(resource_type='workspace', resource_pattern='*', permission=MANAGE)``) or
        a legacy ``workspace_permissions`` MANAGE grant. See
        ``_workspace_admin_workspaces`` for the consolidation rationale.
        """
        with self.ManagedSessionMaker() as session:
            return workspace in self._workspace_admin_workspaces(session, user_id)

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

        Includes grants on the specific resource_type plus workspace-wide grants that
        match every resource type: ``resource_type='workspace'`` (workspace admin) and
        ``resource_type='*'`` (type-wildcard mirroring the legacy
        ``workspace_permissions`` table).

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
                            SqlRolePermission.resource_type.in_(("workspace", "*")),
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
