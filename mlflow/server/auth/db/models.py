from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

from mlflow.server.auth.entities import (
    Role,
    RolePermission,
    User,
    UserRoleAssignment,
)

Base = declarative_base()


class SqlUser(Base):
    __tablename__ = "users"
    id = Column(Integer(), primary_key=True)
    username = Column(String(255), unique=True)
    password_hash = Column(String(255))
    is_admin = Column(Boolean, default=False)
    # Cascade through user_role_assignments so ``session.delete(user)`` cleans
    # up its assignments. Legacy permission tables (experiment_permissions,
    # registered_model_permissions, ...) are retained on disk for rollback
    # by the ``e5f6a7b8c9d0`` migration, but the auth server no longer reads
    # or writes them post-migration; their FK constraints stay enforced at
    # the schema level. delete_user() handles the legacy-row cleanup
    # explicitly when needed.
    user_role_assignments = relationship(
        "SqlUserRoleAssignment",
        backref="user",
        foreign_keys="SqlUserRoleAssignment.user_id",
        cascade="all, delete-orphan",
    )

    def to_mlflow_entity(self):
        return User(
            id_=self.id,
            username=self.username,
            password_hash=self.password_hash,
            is_admin=self.is_admin,
        )


class SqlRole(Base):
    __tablename__ = "roles"

    id = Column(Integer(), primary_key=True)
    name = Column(String(255), nullable=False)
    workspace = Column(String(63), nullable=False)
    description = Column(String(1024), nullable=True)
    permissions = relationship("SqlRolePermission", backref="role", cascade="all, delete-orphan")
    user_assignments = relationship(
        "SqlUserRoleAssignment", backref="role", cascade="all, delete-orphan"
    )
    __table_args__ = (
        UniqueConstraint("workspace", "name", name="unique_workspace_role_name"),
        Index("idx_roles_workspace", "workspace"),
    )

    def to_mlflow_entity(self):
        return Role(
            id_=self.id,
            name=self.name,
            workspace=self.workspace,
            description=self.description,
            permissions=[p.to_mlflow_entity() for p in self.permissions],
        )


class SqlRolePermission(Base):
    __tablename__ = "role_permissions"

    id = Column(Integer(), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    resource_type = Column(String(64), nullable=False)
    resource_pattern = Column(String(255), nullable=False)
    permission = Column(String(255), nullable=False)
    __table_args__ = (
        UniqueConstraint(
            "role_id", "resource_type", "resource_pattern", name="unique_role_resource_perm"
        ),
        Index("idx_role_permissions_role_id", "role_id"),
    )

    def to_mlflow_entity(self):
        return RolePermission(
            id_=self.id,
            role_id=self.role_id,
            resource_type=self.resource_type,
            resource_pattern=self.resource_pattern,
            permission=self.permission,
        )


class SqlUserRoleAssignment(Base):
    __tablename__ = "user_role_assignments"

    id = Column(Integer(), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    __table_args__ = (
        UniqueConstraint("user_id", "role_id", name="unique_user_role"),
        Index("idx_user_role_assignments_user_id", "user_id"),
        Index("idx_user_role_assignments_role_id", "role_id"),
    )

    def to_mlflow_entity(self):
        return UserRoleAssignment(
            id_=self.id,
            user_id=self.user_id,
            role_id=self.role_id,
        )
