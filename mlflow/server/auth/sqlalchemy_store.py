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
            session.delete(user)

    def create_experiment_permission(
        self, experiment_id: str, username: str, permission: str
    ) -> ExperimentPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            try:
                user = self._get_user(session, username=username)
                perm = SqlExperimentPermission(
                    experiment_id=experiment_id, user_id=user.id, permission=permission
                )
                session.add(perm)
                session.flush()
                return perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Experiment permission (experiment_id={experiment_id}, username={username}) "
                    f"already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                )

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
            return perm.to_mlflow_entity()

    def delete_experiment_permission(self, experiment_id: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_experiment_permission(session, experiment_id, username)
            session.delete(perm)

    def delete_workspace_permissions_for_workspace(self, workspace_name: str) -> None:
        with self.ManagedSessionMaker() as session:
            session.query(SqlWorkspacePermission).filter(
                SqlWorkspacePermission.workspace == workspace_name
            ).delete(synchronize_session=False)

    def create_registered_model_permission(
        self, name: str, username: str, permission: str
    ) -> RegisteredModelPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            try:
                user = self._get_user(session, username=username)
                workspace_name = self._get_active_workspace_name()
                perm = SqlRegisteredModelPermission(
                    workspace=workspace_name,
                    name=name,
                    user_id=user.id,
                    permission=permission,
                )
                session.add(perm)
                session.flush()
                return perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    "Registered model permission "
                    f"with workspace={workspace_name}, name={name} and username={username} "
                    f"already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                )

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

    def list_all_registered_model_permissions(
        self, username: str
    ) -> list[RegisteredModelPermission]:
        """
        Cross-workspace variant for callers without an active workspace
        (e.g. the global ``/users/current/permissions`` endpoint backing
        the ``/account`` page). Mirrors ``list_registered_model_permissions``
        but skips the workspace filter so the returned rows span every
        workspace the user has grants in - each row carries its own
        ``workspace`` value, so the caller can still attribute correctly.
        """
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            perms = (
                session
                .query(SqlRegisteredModelPermission)
                .filter(SqlRegisteredModelPermission.user_id == user.id)
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
            return perm.to_mlflow_entity()

    def delete_registered_model_permission(self, name: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_registered_model_permission(session, name, username)
            session.delete(perm)

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

    def list_accessible_workspace_names(self, username: str | None) -> set[str]:
        if username is None:
            return set()

        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)

            # Legacy workspace_permissions grants (pre-RBAC): filter to
            # permissions that actually convey read access.
            legacy_rows: list[SqlWorkspacePermission] = (
                session
                .query(SqlWorkspacePermission)
                .filter(SqlWorkspacePermission.user_id == user.id)
                .all()
            )
            accessible: set[str] = {
                row.workspace for row in legacy_rows if get_permission(row.permission).can_read
            }

            # Role-based: being assigned to any role in a workspace implies
            # membership and therefore visibility of that workspace, even if
            # the role's resource-level permissions don't individually grant
            # read on workspace_permissions.
            role_rows = (
                session
                .query(SqlRole.workspace)
                .join(SqlUserRoleAssignment, SqlRole.id == SqlUserRoleAssignment.role_id)
                .filter(SqlUserRoleAssignment.user_id == user.id)
                .distinct()
                .all()
            )
            accessible.update(w for (w,) in role_rows)

            return accessible

    def get_workspace_permission(self, workspace_name: str, username: str) -> Permission | None:
        """
        Get the **direct** workspace permission for a user — the row in the
        ``workspace_permissions`` table, if any.

        Does NOT include role-based grants. Callers that need the full
        authorization picture should also consult
        ``get_role_workspace_permission`` and ``max_permission``-merge the
        two. See ``mlflow.server.auth.__init__._workspace_permission`` for
        the canonical aggregation.
        """
        with self.ManagedSessionMaker() as session:
            user = self._get_user(session, username=username)
            entity = session.get(SqlWorkspacePermission, (workspace_name, user.id))
            if entity is not None:
                return get_permission(entity.permission)
        return None

    def get_role_workspace_permission(
        self, workspace_name: str, username: str
    ) -> Permission | None:
        """
        Highest **role-based** permission ``username`` has on ``workspace_name``
        where the role grant is workspace-wide (``resource_type='workspace'``,
        ``resource_pattern='*'``). Returns ``None`` when there are no such
        grants.

        Complements ``get_workspace_permission`` — that helper reads the
        ``workspace_permissions`` table directly, this one reads role
        grants. Both are first-class authorization sources; callers that
        need the effective workspace-level permission should max-merge the
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
                    SqlRolePermission.resource_type == "workspace",
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

    def create_scorer_permission(
        self, experiment_id: str, scorer_name: str, username: str, permission: str
    ) -> ScorerPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            try:
                user = self._get_user(session, username=username)
                perm = SqlScorerPermission(
                    experiment_id=experiment_id,
                    scorer_name=scorer_name,
                    user_id=user.id,
                    permission=permission,
                )
                session.add(perm)
                session.flush()
                return perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Scorer permission (experiment_id={experiment_id}, scorer_name={scorer_name}, "
                    f"username={username}) already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                ) from e

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
            return perm.to_mlflow_entity()

    def delete_scorer_permission(self, experiment_id: str, scorer_name: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_scorer_permission(session, experiment_id, scorer_name, username)
            session.delete(perm)

    def delete_scorer_permissions_for_scorer(self, experiment_id: str, scorer_name: str):
        with self.ManagedSessionMaker() as session:
            session.query(SqlScorerPermission).filter(
                SqlScorerPermission.experiment_id == experiment_id,
                SqlScorerPermission.scorer_name == scorer_name,
            ).delete()

    def create_gateway_secret_permission(
        self, secret_id: str, username: str, permission: str
    ) -> GatewaySecretPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            try:
                user = self._get_user(session, username=username)
                perm = SqlGatewaySecretPermission(
                    secret_id=secret_id, user_id=user.id, permission=permission
                )
                session.add(perm)
                session.flush()
                return perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Gateway secret permission (secret_id={secret_id}, username={username}) "
                    f"already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                ) from e

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
            return perm.to_mlflow_entity()

    def delete_gateway_secret_permission(self, secret_id: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_gateway_secret_permission(session, secret_id, username)
            session.delete(perm)

    def delete_gateway_secret_permissions_for_secret(self, secret_id: str):
        with self.ManagedSessionMaker() as session:
            session.query(SqlGatewaySecretPermission).filter(
                SqlGatewaySecretPermission.secret_id == secret_id,
            ).delete()

    def create_gateway_endpoint_permission(
        self, endpoint_id: str, username: str, permission: str
    ) -> GatewayEndpointPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            try:
                user = self._get_user(session, username=username)
                perm = SqlGatewayEndpointPermission(
                    endpoint_id=endpoint_id, user_id=user.id, permission=permission
                )
                session.add(perm)
                session.flush()
                return perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Gateway endpoint permission (endpoint_id={endpoint_id}, username={username}) "
                    f"already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                ) from e

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
            return perm.to_mlflow_entity()

    def delete_gateway_endpoint_permission(self, endpoint_id: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_gateway_endpoint_permission(session, endpoint_id, username)
            session.delete(perm)

    def delete_gateway_endpoint_permissions_for_endpoint(self, endpoint_id: str):
        with self.ManagedSessionMaker() as session:
            session.query(SqlGatewayEndpointPermission).filter(
                SqlGatewayEndpointPermission.endpoint_id == endpoint_id,
            ).delete()

    def create_gateway_model_definition_permission(
        self, model_definition_id: str, username: str, permission: str
    ) -> GatewayModelDefinitionPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            try:
                user = self._get_user(session, username=username)
                perm = SqlGatewayModelDefinitionPermission(
                    model_definition_id=model_definition_id, user_id=user.id, permission=permission
                )
                session.add(perm)
                session.flush()
                return perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Gateway model definition permission "
                    f"(model_definition_id={model_definition_id}, username={username}) "
                    f"already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                ) from e

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
            return perm.to_mlflow_entity()

    def delete_gateway_model_definition_permission(self, model_definition_id: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_gateway_model_definition_permission(
                session, model_definition_id, username
            )
            session.delete(perm)

    def delete_gateway_model_definition_permissions_for_model_definition(
        self, model_definition_id: str
    ):
        with self.ManagedSessionMaker() as session:
            session.query(SqlGatewayModelDefinitionPermission).filter(
                SqlGatewayModelDefinitionPermission.model_definition_id == model_definition_id,
            ).delete()

    # ---- Role CRUD ----

    def create_role(
        self,
        name: str,
        workspace: str,
        description: str | None = None,
    ) -> Role:
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
        # Workspace-scope grants only support the "*" pattern (apply to every resource in
        # the role's workspace). Any other pattern would be silently ignored by the
        # resolver, so reject it up front.
        if resource_type == "workspace" and resource_pattern != "*":
            raise MlflowException.invalid_parameter_value(
                "resource_type='workspace' requires resource_pattern='*'. "
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
                    # Workspace-wide permission — applies to every resource type.
                    if rp.resource_type == "workspace" and rp.resource_pattern == "*":
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
                            SqlRolePermission.resource_type == "workspace",
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
