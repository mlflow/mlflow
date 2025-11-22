from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

from mlflow.server.auth.entities import (
    ExperimentPermission,
    RegisteredModelPermission,
    ScorerPermission,
    User,
    WorkspacePermission,
)

Base = declarative_base()


class SqlUser(Base):
    __tablename__ = "users"
    id = Column(Integer(), primary_key=True)
    username = Column(String(255), unique=True)
    password_hash = Column(String(255))
    is_admin = Column(Boolean, default=False)
    experiment_permissions = relationship("SqlExperimentPermission", backref="users")
    registered_model_permissions = relationship("SqlRegisteredModelPermission", backref="users")
    scorer_permissions = relationship("SqlScorerPermission", backref="users")

    def to_mlflow_entity(self):
        return User(
            id_=self.id,
            username=self.username,
            password_hash=self.password_hash,
            is_admin=self.is_admin,
            experiment_permissions=[p.to_mlflow_entity() for p in self.experiment_permissions],
            registered_model_permissions=[
                p.to_mlflow_entity() for p in self.registered_model_permissions
            ],
            scorer_permissions=[p.to_mlflow_entity() for p in self.scorer_permissions],
        )


class SqlExperimentPermission(Base):
    __tablename__ = "experiment_permissions"
    id = Column(Integer(), primary_key=True)
    experiment_id = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    permission = Column(String(255))
    __table_args__ = (UniqueConstraint("experiment_id", "user_id", name="unique_experiment_user"),)

    def to_mlflow_entity(self):
        return ExperimentPermission(
            experiment_id=self.experiment_id,
            user_id=self.user_id,
            permission=self.permission,
        )


class SqlRegisteredModelPermission(Base):
    __tablename__ = "registered_model_permissions"
    id = Column(Integer(), primary_key=True)
    workspace = Column(String(63), nullable=False)
    name = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    permission = Column(String(255))
    __table_args__ = (
        UniqueConstraint("workspace", "name", "user_id", name="unique_workspace_name_user"),
    )

    def to_mlflow_entity(self):
        return RegisteredModelPermission(
            workspace=self.workspace,
            name=self.name,
            user_id=self.user_id,
            permission=self.permission,
        )


class SqlScorerPermission(Base):
    __tablename__ = "scorer_permissions"
    id = Column(Integer(), primary_key=True)
    experiment_id = Column(String(255), nullable=False)
    scorer_name = Column(String(256), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    permission = Column(String(255))
    __table_args__ = (
        UniqueConstraint("experiment_id", "scorer_name", "user_id", name="unique_scorer_user"),
    )

    def to_mlflow_entity(self):
        return ScorerPermission(
            experiment_id=self.experiment_id,
            scorer_name=self.scorer_name,
            user_id=self.user_id,
            permission=self.permission,
        )


class SqlWorkspacePermission(Base):
    __tablename__ = "workspace_permissions"

    workspace = Column(String(63), nullable=False)
    username = Column(String(255), nullable=False)
    resource_type = Column(String(64), nullable=False)
    permission = Column(String(32), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint(
            "workspace", "username", "resource_type", name="workspace_permissions_pk"
        ),
        Index("idx_workspace_permissions_username", "username"),
        Index("idx_workspace_permissions_workspace", "workspace"),
    )

    def to_mlflow_entity(self):
        return WorkspacePermission(
            workspace=self.workspace,
            username=self.username,
            resource_type=self.resource_type,
            permission=self.permission,
        )
