from typing import List
from sqlalchemy import (
    Column,
    String,
    ForeignKey,
    Integer,
    Boolean,
)
from sqlalchemy.exc import IntegrityError, NoResultFound, MultipleResultsFound
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    INVALID_STATE,
)
from mlflow.server.auth.entities import User, ExperimentPermission, RegisteredModelPermission
from mlflow.server.auth.permissions import _validate_permission
from mlflow.store.db.utils import create_sqlalchemy_engine_with_retry, _get_managed_session_maker
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import _validate_username

Base = declarative_base()


class SqlUser(Base):
    __tablename__ = "users"
    id = Column(Integer(), primary_key=True)
    username = Column(String(255), unique=True)
    password_hash = Column(String(255))
    is_admin = Column(Boolean, default=False)
    experiment_permissions = relationship("SqlExperimentPermission", backref="users")
    registered_model_permissions = relationship("SqlRegisteredModelPermission", backref="users")

    def to_mlflow_entity(self):
        return User(
            id=self.id,
            username=self.username,
            password_hash=self.password_hash,
            is_admin=self.is_admin,
            experiment_permissions=[p.to_mlflow_entity() for p in self.experiment_permissions],
            registered_model_permissions=[
                p.to_mlflow_entity() for p in self.registered_model_permissions
            ],
        )


class SqlExperimentPermission(Base):
    __tablename__ = "experiment_permissions"
    id = Column(Integer(), primary_key=True)
    experiment_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    permission = Column(String(255))

    def to_mlflow_entity(self):
        return ExperimentPermission(
            experiment_id=self.experiment_id,
            user_id=self.user_id,
            permission=self.permission,
        )


class SqlRegisteredModelPermission(Base):
    __tablename__ = "registered_model_permissions"
    id = Column(Integer(), primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    permission = Column(String(255))

    def to_mlflow_entity(self):
        return RegisteredModelPermission(
            name=self.name,
            user_id=self.user_id,
            permission=self.permission,
        )


class SqlAlchemyStore:
    db_uri = None
    db_type = None
    engine = None
    ManagedSessionMaker = None

    def init_db(self, db_uri):
        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.engine = create_sqlalchemy_engine_with_retry(db_uri)
        Base.metadata.create_all(bind=self.engine)
        Base.metadata.bind = self.engine
        SessionMaker = sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = _get_managed_session_maker(SessionMaker, self.db_type)

    def authenticate_user(self, username: str, password: str) -> bool:
        with self.ManagedSessionMaker() as session:
            try:
                user = self._get_user(session, username)
                return check_password_hash(user.password_hash, password)
            except MlflowException:
                return False

    def create_user(self, username: str, password: str, is_admin: bool = False):
        _validate_username(username)
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
                )

    @classmethod
    def _get_user(cls, session, username: str) -> SqlUser:
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

    def list_users(self) -> List[User]:
        with self.ManagedSessionMaker() as session:
            users = session.query(SqlUser).all()
            return [u.to_mlflow_entity() for u in users]

    def create_experiment_permission(
        self, experiment_id: str, user_id: int, permission: str
    ) -> ExperimentPermission:
        _validate_permission(permission)
        with self.ManagedSessionMaker() as session:
            try:
                perm = SqlExperimentPermission(
                    experiment_id=experiment_id, user_id=user_id, permission=permission
                )
                session.add(perm)
                session.flush()
                return perm.to_mlflow_entity()
            except IntegrityError as e:
                raise MlflowException(
                    f"Experiment permission (experiment_id={experiment_id}, user_id={user_id}, "
                    f"permission={permission}) already exists. Error: {e}",
                    RESOURCE_ALREADY_EXISTS,
                )

    @classmethod
    def _get_experiment_permission(
        cls, session, experiment_id: str, user_id: int
    ) -> SqlExperimentPermission:
        try:
            return (
                session.query(SqlExperimentPermission)
                .filter(SqlExperimentPermission.experiment_id == experiment_id)
                .filter(SqlExperimentPermission.user_id == user_id)
                .one()
            )
        except NoResultFound:
            raise MlflowException(
                f"Experiment permission with experiment_id={experiment_id} and user_id={user_id} not found",
                RESOURCE_DOES_NOT_EXIST,
            )
        except MultipleResultsFound:
            raise MlflowException(
                f"Found multiple experiment permissions with experiment_id={experiment_id} and user_id={user_id}",
                INVALID_STATE,
            )

    def get_experiment_permission(self, experiment_id: str, user_id: int) -> ExperimentPermission:
        with self.ManagedSessionMaker() as session:
            return self._get_experiment_permission(
                session, experiment_id, user_id
            ).to_mlflow_entity()

    def list_experiment_permissions(self, user_id: int) -> List[ExperimentPermission]:
        with self.ManagedSessionMaker() as session:
            perms = (
                session.query(SqlExperimentPermission)
                .filter(SqlExperimentPermission.user_id == user_id)
                .all()
            )
            return [p.to_mlflow_entity() for p in perms]

    def update_experiment_permission(
        self, experiment_id: str, user_id: int, permission: str
    ) -> ExperimentPermission:
        with self.ManagedSessionMaker() as session:
            perm = self._get_experiment_permission(session, experiment_id, user_id)
            perm.permission = permission
            return perm.to_mlflow_entity()

    def delete_experiment_permission(self, experiment_id: str, user_id: int):
        with self.ManagedSessionMaker() as session:
            perm = self._get_experiment_permission(session, experiment_id, user_id)
            session.delete(perm)
