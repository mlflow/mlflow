from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

from mlflow.server.auth.entities import (
    ExperimentPermission,
    RegisteredModelPermission,
    ScorerPermission,
    User,
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
    name = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    permission = Column(String(255))
    __table_args__ = (UniqueConstraint("name", "user_id", name="unique_name_user"),)

    def to_mlflow_entity(self):
        return RegisteredModelPermission(
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
