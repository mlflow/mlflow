from sqlalchemy.exc import IntegrityError, MultipleResultsFound, NoResultFound
from sqlalchemy.orm import sessionmaker
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
    SqlRegisteredModelPermission,
    SqlScorerPermission,
    SqlUser,
    SqlWorkspacePermission,
)
from mlflow.server.auth.entities import (
    ExperimentPermission,
    RegisteredModelPermission,
    ScorerPermission,
    User,
    WorkspacePermission,
)
from mlflow.server.auth.permissions import Permission, _validate_permission, get_permission
from mlflow.store.db.utils import _get_managed_session_maker, create_sqlalchemy_engine_with_retry
from mlflow.tracking._workspace import context as workspace_context
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import _validate_password, _validate_username
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


class SqlAlchemyStore:
    _WORKSPACE_RESOURCE_TYPES = {"*", "experiments", "registered_models"}

    @staticmethod
    def _workspaces_enabled() -> bool:
        return bool(MLFLOW_ENABLE_WORKSPACES.get())

    @classmethod
    def _get_active_workspace_name(cls) -> str:
        if not cls._workspaces_enabled():
            return DEFAULT_WORKSPACE_NAME

        workspace_name = workspace_context.get_current_workspace()
        if workspace_name:
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
                session.query(SqlExperimentPermission)
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
                session.query(SqlExperimentPermission)
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

    def supports_workspaces(self) -> bool:
        return True

    def _validate_workspace_resource_type(self, resource_type: str) -> None:
        if resource_type not in self._WORKSPACE_RESOURCE_TYPES:
            raise MlflowException.invalid_parameter_value(
                f"Invalid resource type '{resource_type}'. Valid resource types are: "
                f"{', '.join(sorted(self._WORKSPACE_RESOURCE_TYPES))}"
            )

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
                session.query(SqlRegisteredModelPermission)
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
                session.query(SqlRegisteredModelPermission)
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
            return perm.to_mlflow_entity()

    def delete_registered_model_permission(self, name: str, username: str):
        with self.ManagedSessionMaker() as session:
            perm = self._get_registered_model_permission(session, name, username)
            session.delete(perm)

    def rename_registered_model_permissions(self, old_name: str, new_name: str):
        with self.ManagedSessionMaker() as session:
            workspace_name = self._get_active_workspace_name()
            perms = (
                session.query(SqlRegisteredModelPermission)
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
                session.query(SqlWorkspacePermission)
                .filter(SqlWorkspacePermission.workspace == workspace_name)
                .all()
            )
            return [row.to_mlflow_entity() for row in rows]

    def list_user_workspace_permissions(self, username: str) -> list[WorkspacePermission]:
        with self.ManagedSessionMaker() as session:
            rows = (
                session.query(SqlWorkspacePermission)
                .filter(SqlWorkspacePermission.username.in_([username, "*"]))
                .all()
            )
            return [row.to_mlflow_entity() for row in rows]

    def set_workspace_permission(
        self, workspace_name: str, username: str, resource_type: str, permission: str
    ) -> WorkspacePermission:
        self._validate_workspace_resource_type(resource_type)
        _validate_permission(permission)

        with self.ManagedSessionMaker() as session:
            entity = session.get(
                SqlWorkspacePermission,
                (workspace_name, username, resource_type),
            )
            if entity is None:
                entity = SqlWorkspacePermission(
                    workspace=workspace_name,
                    username=username,
                    resource_type=resource_type,
                    permission=permission,
                )
                session.add(entity)
            else:
                entity.permission = permission
            session.flush()
            return entity.to_mlflow_entity()

    def delete_workspace_permission(
        self, workspace_name: str, username: str, resource_type: str
    ) -> None:
        self._validate_workspace_resource_type(resource_type)
        with self.ManagedSessionMaker() as session:
            entity = session.get(
                SqlWorkspacePermission,
                (workspace_name, username, resource_type),
            )
            if entity is None:
                raise MlflowException(
                    (
                        "Workspace permission does not exist for "
                        f"workspace='{workspace_name}', username='{username}', "
                        f"resource_type='{resource_type}'"
                    ),
                    RESOURCE_DOES_NOT_EXIST,
                )
            session.delete(entity)

    def list_accessible_workspace_names(self, username: str | None) -> set[str]:
        if username is None:
            return set()

        with self.ManagedSessionMaker() as session:
            rows: list[SqlWorkspacePermission] = (
                session.query(SqlWorkspacePermission)
                .filter(SqlWorkspacePermission.username.in_([username, "*"]))
                .all()
            )
            accessible: set[str] = set()
            for row in rows:
                permission = row.permission
                if not permission:
                    continue
                _validate_permission(permission)
                if get_permission(permission).can_read:
                    accessible.add(row.workspace)
            return accessible

    def get_workspace_permission(
        self, workspace_name: str, username: str, resource_type: str
    ) -> Permission | None:
        """
        Get the most specific workspace permission for a user and resource type.

        Permission precedence (highest to lowest):
        1. Exact user + exact resource type
        2. Exact user + wildcard resource type (*)
        3. Wildcard user (*) + exact resource type
        4. Wildcard user (*) + wildcard resource type (*)
        """
        self._validate_workspace_resource_type(resource_type)

        with self.ManagedSessionMaker() as session:
            # Query all matching permissions and build lookup dictionary
            rows = (
                session.query(SqlWorkspacePermission)
                .filter(
                    SqlWorkspacePermission.workspace == workspace_name,
                    SqlWorkspacePermission.username.in_([username, "*"]),
                    SqlWorkspacePermission.resource_type.in_([resource_type, "*"]),
                )
                .all()
            )

            lookup = {(row.username, row.resource_type): row.permission for row in rows}

        # Check permissions in order of specificity
        # This ensures we always return the most specific permission first
        priority_checks = [
            (username, resource_type),  # Most specific: exact user + exact resource
            (username, "*"),  # User-specific but any resource
            ("*", resource_type),  # Any user but specific resource
            ("*", "*"),  # Least specific: any user + any resource
        ]

        for username_key, resource_key in priority_checks:
            if (username_key, resource_key) in lookup:
                permission_value = lookup[(username_key, resource_key)]
                _validate_permission(permission_value)
                return get_permission(permission_value)

        return None

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
                session.query(SqlScorerPermission)
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
                session.query(SqlScorerPermission)
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
