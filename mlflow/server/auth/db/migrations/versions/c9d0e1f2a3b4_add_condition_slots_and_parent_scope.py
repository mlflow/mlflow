"""Add condition slots and container scope to mutation_conditions

Revision ID: c9d0e1f2a3b4
Revises: b7c8d9e0f1a2
Create Date: 2026-10-03 18:45:00.000000

Widens ``mutation_conditions`` from one object per ``(role, resource_type)`` to
up to ``MAX_CONDITIONS_PER_ROLE_TYPE``, each optionally scoped to an exact
direct parent.

Three columns and three constraints:

- ``condition_slot`` with a range CHECK, plus a UNIQUE on
  ``(role_id, resource_type, condition_slot)`` replacing the old UNIQUE on
  ``(role_id, resource_type)``. Together these bound the per-role-and-type count
  without a count-then-insert race: a racing inserter loses the UNIQUE and
  retries a different slot rather than overshooting the limit.
- ``container_resource_type`` and ``container_resource_pattern``, both NOT NULL
  with defaults of ``'workspace'`` and ``'*'``, plus a CHECK that a ``workspace``
  container carries only the wildcard. Every condition has a container, which is
  what lets a top-level type be addressed at all: it sits in the workspace. A
  wildcard container on a real type is normalised to ``workspace`` before it
  reaches the database, so a non-workspace row always names a concrete id.
- a lookup index on ``(role_id, resource_type, container_resource_type,
  container_resource_pattern)`` so the gate's container predicate runs in SQL.
  (``resource_pattern`` is deliberately NOT in it: the gate matches that axis in
  Python, because a cascade's child ids are unknown when the query runs.) Without it a
  role at the limit would transfer every condition on every request, turning a
  storage bound into a per-request cost.

Behaviour-neutral on upgrade. Existing rows -- there should be none, the feature
is unreleased, but the migration does not assume that -- take slot 1 and the
workspace container, which is exactly their current meaning. The slot's ``server_default``
exists only to satisfy NOT NULL during that backfill and is dropped afterwards:
an INSERT that omits the slot is a store bug and should fail, not silently land
in slot 1.

``downgrade`` is lossy where upgrade was not. Conditions beyond slot 1, and any
container scope, cannot be represented by the old schema -- so rather than silently
broadening a scoped restriction to the whole workspace, or silently dropping
restrictions and leaving a role *less* constrained than the admin wrote, it
refuses when either exists. A deployment that genuinely wants to downgrade must
delete those objects first and so make the loss explicit.
"""

import sqlalchemy as sa
from alembic import op

from mlflow.server.auth.conditions import MAX_CONDITIONS_PER_ROLE_TYPE

# revision identifiers, used by Alembic.
revision = "c9d0e1f2a3b4"
down_revision = "b7c8d9e0f1a2"
branch_labels = None
depends_on = None

_SLOT_RANGE_CHECK = f"condition_slot BETWEEN 1 AND {MAX_CONDITIONS_PER_ROLE_TYPE}"
_CONTAINER_WORKSPACE_CHECK = (
    "container_resource_type <> 'workspace' OR container_resource_pattern = '*'"
)


def upgrade() -> None:
    # Batch mode: SQLite cannot drop a named constraint in place, so Alembic
    # rebuilds the table. Harmless on the other backends, which get plain ALTERs.
    with op.batch_alter_table("mutation_conditions") as batch_op:
        batch_op.add_column(
            sa.Column("condition_slot", sa.SmallInteger(), nullable=False, server_default="1")
        )
        batch_op.add_column(
            sa.Column(
                "container_resource_type",
                sa.String(length=64),
                nullable=False,
                server_default="workspace",
            )
        )
        batch_op.add_column(
            sa.Column(
                "container_resource_pattern",
                sa.String(length=255),
                nullable=False,
                server_default="*",
            )
        )
        batch_op.drop_constraint("unique_role_resource_type", type_="unique")
        batch_op.create_unique_constraint(
            "unique_role_resource_type_slot", ["role_id", "resource_type", "condition_slot"]
        )
        batch_op.create_check_constraint("ck_mutation_conditions_slot_range", _SLOT_RANGE_CHECK)
        batch_op.create_check_constraint(
            "ck_mutation_conditions_container_workspace", _CONTAINER_WORKSPACE_CHECK
        )

    # Drop the backfill default now that every existing row has a slot.
    with op.batch_alter_table("mutation_conditions") as batch_op:
        batch_op.alter_column("condition_slot", server_default=None)

    op.create_index(
        "idx_mutation_conditions_lookup",
        "mutation_conditions",
        ["role_id", "resource_type", "container_resource_type", "container_resource_pattern"],
        unique=False,
    )


def downgrade() -> None:
    # Refuse rather than silently change what a condition means. See the module
    # docstring: the old schema can represent neither a second slot nor a parent
    # scope, and both possible coercions are wrong in a security-relevant way.
    connection = op.get_bind()
    blocking = connection.execute(
        sa.text(
            "SELECT COUNT(*) FROM mutation_conditions "
            "WHERE condition_slot <> 1 OR container_resource_type <> 'workspace'"
        )
    ).scalar()
    if blocking:
        raise RuntimeError(
            f"Cannot downgrade: {blocking} mutation condition(s) use a slot other than 1 or a "
            f"container scope, and the previous schema can represent neither. Downgrading would "
            f"either broaden a container-scoped restriction to the whole workspace or drop "
            f"restrictions entirely, leaving roles less constrained than configured. Delete "
            f"those conditions first if the loss is intended."
        )

    op.drop_index("idx_mutation_conditions_lookup", table_name="mutation_conditions")
    with op.batch_alter_table("mutation_conditions") as batch_op:
        batch_op.drop_constraint("ck_mutation_conditions_container_workspace", type_="check")
        batch_op.drop_constraint("ck_mutation_conditions_slot_range", type_="check")
        batch_op.drop_constraint("unique_role_resource_type_slot", type_="unique")
        batch_op.create_unique_constraint("unique_role_resource_type", ["role_id", "resource_type"])
        batch_op.drop_column("container_resource_pattern")
        batch_op.drop_column("container_resource_type")
        batch_op.drop_column("condition_slot")
