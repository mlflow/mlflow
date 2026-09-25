from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

from mlflow.server.auth.entities import (
    MutationConditions,
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
    # Conditions are per (role, resource_type) and meaningless without the role,
    # so they cascade exactly like permissions do. The FK also declares
    # ON DELETE CASCADE for the DB-level path (raw SQL, or a dialect with FK
    # enforcement on), but the ORM cascade is what ``delete_role`` relies on.
    mutation_conditions = relationship(
        "SqlMutationConditions", backref="role", cascade="all, delete-orphan"
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


class SqlMutationConditions(Base):
    """Condition-based access control: two optional filters per ``(role, resource_type)``
    that gate create/mutation operations only, never reads.

    ``value_condition`` (the RFC's *value*, this codebase's **request condition**)
    constrains what values may be set, and is evaluated against the request body.
    ``target_condition`` (the RFC's *target*, the **resource condition**) constrains
    which existing resources may be mutated, and is evaluated against the resource's
    current state.

    Both are nullable: a role may carry one, both, or neither. Each is stored as the
    filter string the admin authored, in MLflow's search-filter grammar, and parsed
    at read time (``mlflow.server.auth.conditions``). Storing the string rather than a
    decomposed form keeps ``list`` round-trippable and the schema stable; the RFC
    leaves the representation open.

    Conditions only ever **subtract** from what grants allow -- an empty table is
    exactly today's behaviour, and a condition never confers access.
    """

    __tablename__ = "mutation_conditions"

    id = Column(Integer(), primary_key=True)
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="CASCADE"), nullable=False)
    # 64 to match SqlRolePermission.resource_type -- the same vocabulary, so the
    # same width. (The RFC says 255; matching the sibling column matters more.)
    resource_type = Column(String(64), nullable=False)
    # Text, not String(n): a filter string has no meaningful length bound beyond
    # the clause-count limit the parser enforces.
    value_condition = Column(Text, nullable=True)
    target_condition = Column(Text, nullable=True)
    __table_args__ = (
        # At most one of each condition per (role, resource_type), per the RFC.
        UniqueConstraint("role_id", "resource_type", name="unique_role_resource_type"),
        Index("idx_mutation_conditions_role_id", "role_id"),
    )

    def to_mlflow_entity(self):
        return MutationConditions(
            id_=self.id,
            role_id=self.role_id,
            resource_type=self.resource_type,
            value_condition=self.value_condition,
            target_condition=self.target_condition,
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
