"""Add conditions table for condition-based access control

Revision ID: 262a5e73fb1a
Revises: f1a2b3c4d5e6
Create Date: 2026-10-09 01:30:00.000000

Adds the ``conditions`` table: up to ``MAX_CONDITIONS_PER_ROLE_TYPE``
typed filters per ``(role, resource_type)``, each addressing the resources it
governs and optionally scoped to one exact container. Conditions gate create and
mutation operations only, never reads.

A ``VALUE`` condition constrains the values a request may set and is evaluated
against the request body. A ``TARGET`` condition constrains which existing
resources may be mutated and is evaluated against current state. ``filter`` is
NOT NULL -- a row always carries exactly one filter, and a role that should not
constrain a type stores no row of that type.

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

``condition_slot`` says which of the role's conditions OF THAT TYPE a row is. It
carries no ordering meaning -- same-type conditions OR, and OR is commutative, so
there is nothing to order. Its whole job is to make the per-``(role, type,
condition_type)`` bound race-safe: the UNIQUE on ``(role_id, resource_type,
condition_type, condition_slot)`` and the range CHECK together enforce
"at most ``MAX_CONDITIONS_PER_ROLE_TYPE``" without a count-then-insert race,
because a racing inserter loses the UNIQUE and retries a different slot rather
than overshooting the limit. It has no ``server_default``: an INSERT that omits
the slot is a store bug and should fail rather than silently land in slot 1.

``condition_type`` says which question a row's filter answers -- ``VALUE`` reads the
request, ``TARGET`` reads the target's current state -- and so which namespace the
filter is parsed against. Rows sharing a type are alternatives combined with OR; the
two types combine with AND. Each type has its own slot space, which is why the
UNIQUE includes it. A CHECK pins the column to the known vocabulary, because an
unrecognised type would store cleanly and then be silently skipped by the evaluator,
failing open. A future search condition is one more value here, not a new column.

The lookup index covers how the gate resolves conditions -- by role, target type,
condition type, and either no container scope or the exact resolved container -- so
that predicate runs in SQL and row volume stays proportional to the conditions that
*apply* rather than to the number stored. ``condition_type`` is in it because the gate
loads one type at a time: the value side and the target side are separate
disjunctions. ``resource_pattern`` is deliberately absent from
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
        "conditions",
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
        # Which question this row's filter answers: VALUE reads the request, TARGET reads
        # the target's current state. A CHECK below keeps the column to the known
        # vocabulary, because an unrecognised type would store cleanly and then be
        # silently skipped by the evaluator -- failing open.
        sa.Column("condition_type", sa.String(length=16), nullable=False),
        # Text, not String(n): a filter string has no meaningful length bound beyond the
        # clause-count limit the parser enforces. NOT NULL: a row with no filter would
        # answer its question with "anything", which under OR is an unconditional pass
        # that nullifies every sibling of its type.
        sa.Column("filter", sa.Text(), nullable=False),
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
            "role_id",
            "resource_type",
            "condition_type",
            "condition_slot",
            name="unique_role_resource_type_slot",
        ),
        sa.CheckConstraint(
            f"condition_slot BETWEEN 1 AND {_SLOT_RANGE_LIMIT}",
            name="ck_conditions_slot_range",
        ),
        sa.CheckConstraint(
            "condition_type IN ('VALUE', 'TARGET')",
            name="ck_conditions_type",
        ),
        # The workspace container is named by the role's own workspace column rather than
        # by a pattern, so it accepts only the wildcard -- the grain a workspace grant has.
        sa.CheckConstraint(
            "container_resource_type <> 'workspace' OR container_resource_pattern = '*'",
            name="ck_conditions_container_workspace",
        ),
    )
    op.create_index(
        "idx_conditions_lookup",
        "conditions",
        [
            "role_id",
            "resource_type",
            "condition_type",
            "container_resource_type",
            "container_resource_pattern",
        ],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_conditions_lookup", table_name="conditions")
    op.drop_table("conditions")
