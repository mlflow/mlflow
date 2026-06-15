"""add name_key to review_queues for case-insensitive name uniqueness

Revision ID: e2c4a6b80d15
Revises: c5d9e1f3a7b2

Create Date: 2026-06-16 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e2c4a6b80d15"
down_revision = "c5d9e1f3a7b2"
branch_labels = None
depends_on = None


def upgrade():
    # Custom-queue names were stored case-preserved while user-queue names were
    # lowercased, and the unique key was the raw `name` — so `Foo`/`foo` could
    # coexist on a case-sensitive backend (and the same create behaved differently
    # on a case-insensitive one). Add a case-folded `name_key` and move the
    # uniqueness to (experiment_id, name_key) so names are unique
    # case-insensitively across both queue types on every backend; `name` keeps
    # the display casing.
    bind = op.get_bind()
    review_queues = sa.table(
        "review_queues",
        sa.column("queue_id", sa.String),
        sa.column("experiment_id", sa.Integer),
        sa.column("name", sa.String),
        sa.column("name_key", sa.String),
    )
    rows = bind.execute(
        sa.select(review_queues.c.queue_id, review_queues.c.experiment_id, review_queues.c.name)
    ).fetchall()

    # Fail fast — before any schema change — if existing rows already collide
    # case-insensitively (only possible on a case-sensitive backend, since the old
    # raw-`name` constraint blocked exact-case dupes). Adding the new constraint
    # would otherwise abort mid-migration with an opaque IntegrityError; surface an
    # actionable message naming the pairs instead.
    seen: dict[tuple[int, str], str] = {}
    collisions = []
    for _, experiment_id, name in rows:
        key = (experiment_id, name.lower())
        if key in seen:
            collisions.append((experiment_id, seen[key], name))
        else:
            seen[key] = name
    if collisions:
        details = "; ".join(f"experiment {e}: {a!r} vs {b!r}" for e, a, b in collisions)
        raise RuntimeError(
            "Cannot upgrade review_queues: case-insensitively duplicate queue names exist and "
            "would violate the new unique (experiment_id, name_key) constraint. Rename or delete "
            f"one queue in each pair, then re-run the migration. Collisions: {details}"
        )

    # Add the column nullable, backfill in Python (so `name_key` matches exactly
    # what the store computes — `name.lower()` — independent of any DB `lower()` /
    # collation behavior), then enforce NOT NULL and swap the constraint.
    with op.batch_alter_table("review_queues", schema=None) as batch_op:
        batch_op.add_column(sa.Column("name_key", sa.String(length=250), nullable=True))
    for queue_id, _, name in rows:
        bind.execute(
            review_queues
            .update()
            .where(review_queues.c.queue_id == queue_id)
            .values(name_key=name.lower())
        )
    with op.batch_alter_table("review_queues", schema=None) as batch_op:
        batch_op.alter_column("name_key", existing_type=sa.String(length=250), nullable=False)
        batch_op.drop_constraint("uq_review_queues_experiment_name", type_="unique")
        batch_op.create_unique_constraint(
            "uq_review_queues_experiment_name_key", ["experiment_id", "name_key"]
        )


def downgrade():
    with op.batch_alter_table("review_queues", schema=None) as batch_op:
        batch_op.drop_constraint("uq_review_queues_experiment_name_key", type_="unique")
        batch_op.create_unique_constraint(
            "uq_review_queues_experiment_name", ["experiment_id", "name"]
        )
        batch_op.drop_column("name_key")
