"""Add mutation_conditions table for condition-based access control

Revision ID: 262a5e73fb1a
Revises: f1a2b3c4d5e6
Create Date: 2026-10-09 01:30:00.000000

Adds the ``mutation_conditions`` table: up to ``MAX_CONDITIONS_PER_ROLE_TYPE``
filter pairs per ``(role, resource_type)``, each addressing the resources it
governs and optionally scoped to one exact container. Conditions gate create and
mutation operations only, never reads.

``value_condition`` constrains the values a request may set and is evaluated
against the request body. ``target_condition`` constrains which existing
resources may be mutated and is evaluated against current state. Both are
nullable -- a row may carry one, both, or neither.

Two scope axes, both NOT NULL with meaningful defaults so that "everything" has
exactly one representation; a nullable column would let NULL and ``'*'`` both
mean all, and the gate's check would then depend on which an admin happened to
type:

- ``resource_pattern`` -- which resources of the type, ``'*'`` or one id, at the
  grain the type's grants use;
- ``container_resource_type`` / ``container_resource_pattern`` -- within which
  container, ``'workspace'``/``'*'`` for no narrowing, otherwise the type's
  declared parent and one of its ids. A wildcard container on a real type is
  normalised to ``workspace`` before it reaches the database, so a non-workspace
  row always names a concrete id.

``condition_slot`` says which of the role's conditions for that type a row is. It
carries no ordering meaning -- conditions all AND, so there is nothing to order.
Its whole job is to make the per-``(role, type)`` bound race-safe: the UNIQUE on
``(role_id, resource_type, condition_slot)`` and the range CHECK together enforce
"at most ``MAX_CONDITIONS_PER_ROLE_TYPE``" without a count-then-insert race,
because a racing inserter loses the UNIQUE and retries a different slot rather
than overshooting the limit. It has no ``server_default``: an INSERT that omits
the slot is a store bug and should fail rather than silently land in slot 1.

The lookup index covers how the gate resolves conditions -- by role, target type,
and either no container scope or the exact resolved container -- so that predicate
runs in SQL and row volume stays proportional to the conditions that *apply*
rather than to the number stored. ``resource_pattern`` is deliberately absent from
it: the gate matches that axis in Python, because a cascade's child ids are not
known when the query runs.

The table starts empty, and an empty table is exactly the pre-existing behaviour
-- conditions only ever subtract from what grants allow. So this migration is
behaviour-neutral on upgrade, and ``downgrade`` is lossy only in that it drops
conditions an admin authored after upgrading; it cannot grant anyone access they
did not already have.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "262a5e73fb1a"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None

# Frozen at the value ``MAX_CONDITIONS_PER_ROLE_TYPE`` held when this revision was
# written, deliberately NOT imported from it. A migration has to describe the schema it
# produced: importing the live constant would make a fresh database replaying this
# revision build a different CHECK than every database already migrated, and would leave
# a later revision unable to assume the bound it is widening from. Raising the limit means
# adding a revision that alters this constraint, not editing this line.
_SLOT_RANGE_LIMIT = 100


def upgrade() -> None:
    op.create_table(
        "mutation_conditions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "role_id",
            sa.Integer(),
            sa.ForeignKey("roles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # 64 to match role_permissions.resource_type -- the same vocabulary, so the same
        # width. Likewise 255 for the patterns: the same identifiers.
        sa.Column("resource_type", sa.String(length=64), nullable=False),
        sa.Column("resource_pattern", sa.String(length=255), nullable=False, server_default="*"),
        # Text, not String(n): a filter string has no meaningful length bound beyond the
        # clause-count limit the parser enforces.
        sa.Column("value_condition", sa.Text(), nullable=True),
        sa.Column("target_condition", sa.Text(), nullable=True),
        sa.Column("condition_slot", sa.SmallInteger(), nullable=False),
        sa.Column(
            "container_resource_type",
            sa.String(length=64),
            nullable=False,
            server_default="workspace",
        ),
        sa.Column(
            "container_resource_pattern",
            sa.String(length=255),
            nullable=False,
            server_default="*",
        ),
        sa.UniqueConstraint(
            "role_id", "resource_type", "condition_slot", name="unique_role_resource_type_slot"
        ),
        sa.CheckConstraint(
            f"condition_slot BETWEEN 1 AND {_SLOT_RANGE_LIMIT}",
            name="ck_mutation_conditions_slot_range",
        ),
        # The workspace container is named by the role's own workspace column rather than
        # by a pattern, so it accepts only the wildcard -- the grain a workspace grant has.
        sa.CheckConstraint(
            "container_resource_type <> 'workspace' OR container_resource_pattern = '*'",
            name="ck_mutation_conditions_container_workspace",
        ),
    )
    op.create_index(
        "idx_mutation_conditions_lookup",
        "mutation_conditions",
        ["role_id", "resource_type", "container_resource_type", "container_resource_pattern"],
        unique=False,
    )
    op.create_index(
        "idx_mutation_conditions_role_id", "mutation_conditions", ["role_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("idx_mutation_conditions_role_id", table_name="mutation_conditions")
    op.drop_index("idx_mutation_conditions_lookup", table_name="mutation_conditions")
    op.drop_table("mutation_conditions")
