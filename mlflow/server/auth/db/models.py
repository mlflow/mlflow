from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

from mlflow.server.auth.conditions import MAX_CONDITIONS_PER_ROLE_TYPE
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
    # Which of the role's conditions for this resource type this row is. The slot is
    # allocated by the store, never supplied by the caller, and carries no ordering
    # meaning: conditions all AND, so there is nothing to order. Its whole job is to
    # make the per-(role, type) bound race-safe -- UNIQUE below plus the CHECK range
    # enforce "at most MAX_CONDITIONS_PER_ROLE_TYPE" without a count-then-insert race.
    #
    # Deliberately no server_default: an INSERT that omits the slot is a store bug, and
    # should fail rather than silently land in slot 1 and collide.
    condition_slot = Column(SmallInteger, nullable=False)
    # The two scope axes, both NOT NULL with meaningful defaults so that "everything" has
    # exactly one representation -- a nullable column would let NULL and '*' both mean all,
    # and the gate's check would then depend on which an admin happened to type.
    #
    # resource_pattern: which resources of this type. '*' or one id, at the grain the type's
    # grants use (permissions.TYPE) -- so the four top-level conditionable types take an id
    # and the six sub-resources are wildcard-only.
    #
    # container_*: within which container. 'workspace'/'*' is no narrowing; otherwise the
    # type's declared parent and one of its ids. A wildcard container is normalised to
    # 'workspace' on write, so a non-workspace container always carries a concrete id --
    # which is what keeps the loader's predicate to two branches.
    #
    # Widths match SqlRolePermission: the same vocabulary and the same identifiers.
    resource_pattern = Column(String(255), nullable=False, server_default="*")
    container_resource_type = Column(String(64), nullable=False, server_default="workspace")
    container_resource_pattern = Column(String(255), nullable=False, server_default="*")
    __table_args__ = (
        # One row per slot. Replaces the single-object UniqueConstraint on
        # (role_id, resource_type): a role may now hold up to
        # MAX_CONDITIONS_PER_ROLE_TYPE conditions for each type, which is what makes
        # per-parent scoping expressible -- one object per governed parent.
        UniqueConstraint(
            "role_id", "resource_type", "condition_slot", name="unique_role_resource_type_slot"
        ),
        CheckConstraint(
            f"condition_slot BETWEEN 1 AND {MAX_CONDITIONS_PER_ROLE_TYPE}",
            name="ck_mutation_conditions_slot_range",
        ),
        # The workspace container is named by the role's own workspace column, not by a
        # pattern, so it accepts only the wildcard -- the same grain a workspace grant has.
        # This replaces the old parent-pair CHECK, which existed only because the scope was
        # two nullable columns that had to be set together; with NOT NULL defaults there is
        # no half-set state left to forbid.
        CheckConstraint(
            "container_resource_type <> 'workspace' OR container_resource_pattern = '*'",
            name="ck_mutation_conditions_container_workspace",
        ),
        # The lookup index. The gate resolves conditions by role, target type, and
        # either no parent scope or the exact resolved parent -- so the predicate runs
        # in SQL and row volume stays proportional to the conditions that *apply*, not
        # to the number stored. Without the parent columns in the index, a role holding
        # the full MAX_CONDITIONS_PER_ROLE_TYPE would transfer all of them on every
        # request, turning a storage bound into a per-request cost.
        Index(
            "idx_mutation_conditions_lookup",
            "role_id",
            "resource_type",
            "container_resource_type",
            "container_resource_pattern",
        ),
    )

    def to_mlflow_entity(self):
        return MutationConditions(
            id_=self.id,
            role_id=self.role_id,
            resource_type=self.resource_type,
            value_condition=self.value_condition,
            target_condition=self.target_condition,
            condition_slot=self.condition_slot,
            resource_pattern=self.resource_pattern,
            container_resource_type=self.container_resource_type,
            container_resource_pattern=self.container_resource_pattern,
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
