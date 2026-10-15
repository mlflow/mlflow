"""Extend SQL Issues with observation, provenance, metadata and indexed facets."""

import hashlib
import json

import sqlalchemy as sa
from alembic import op

revision = "8b6ad64cb8f5"
down_revision = "dc11669786a5"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("issues") as batch_op:
        batch_op.add_column(sa.Column("last_seen_timestamp", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("source_type", sa.String(50), nullable=True))
        batch_op.add_column(sa.Column("source_id", sa.String(255), nullable=True))
        batch_op.add_column(sa.Column("last_updated_by", sa.String(255), nullable=True))
        batch_op.add_column(sa.Column("metadata", sa.Text(), nullable=True))
        batch_op.create_index(
            "index_issues_experiment_last_seen", ["experiment_id", "last_seen_timestamp"]
        )
        batch_op.create_index(
            "index_issues_experiment_updated", ["experiment_id", "last_updated_timestamp"]
        )
    op.create_table(
        "issue_tags",
        sa.Column(
            "issue_id",
            sa.String(36),
            sa.ForeignKey("issues.issue_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("key", sa.String(250), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
    )
    op.create_index("index_issue_tags_key", "issue_tags", ["key"])
    op.create_table(
        "issue_categories",
        sa.Column(
            "issue_id",
            sa.String(36),
            sa.ForeignKey("issues.issue_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("category_hash", sa.String(64), primary_key=True),
        sa.Column("category", sa.Text(), nullable=False),
    )
    op.create_index("index_issue_categories_hash", "issue_categories", ["category_hash"])
    issues = sa.table(
        "issues",
        sa.column("issue_id"),
        sa.column("categories"),
        sa.column("created_timestamp"),
        sa.column("last_seen_timestamp"),
        sa.column("source_run_id"),
        sa.column("source_type"),
        sa.column("source_id"),
    )
    conn = op.get_bind()
    conn.execute(issues.update().values(last_seen_timestamp=issues.c.created_timestamp))
    conn.execute(
        issues
        .update()
        .where(issues.c.source_run_id.is_not(None))
        .values(source_type="run", source_id=issues.c.source_run_id)
    )
    categories = sa.table(
        "issue_categories", sa.column("issue_id"), sa.column("category_hash"), sa.column("category")
    )
    # Read in batches; keep the original JSON column for older readers.
    rows = conn.execute(
        sa.select(issues.c.issue_id, issues.c.categories).where(issues.c.categories.is_not(None))
    )
    while batch := rows.fetchmany(500):
        entries = [
            {
                "issue_id": row.issue_id,
                "category": category,
                "category_hash": hashlib.sha256(category.encode()).hexdigest(),
            }
            for row in batch
            for category in set(json.loads(row.categories))
        ]
        if entries:
            conn.execute(categories.insert(), entries)


def downgrade():
    raise NotImplementedError("Downgrading the Issue schema would discard metadata and provenance")
