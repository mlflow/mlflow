"""Extend SQL Issues with observation, metadata and indexed facets."""

import hashlib
import json

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.mysql import LONGTEXT

revision = "8b6ad64cb8f5"
down_revision = "e7d1f4b2a9c6"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("issues") as batch_op:
        batch_op.add_column(sa.Column("last_seen_timestamp", sa.BigInteger(), nullable=True))
        batch_op.add_column(sa.Column("last_updated_by", sa.String(255), nullable=True))
        batch_op.add_column(
            sa.Column("metadata", sa.Text().with_variant(LONGTEXT(), "mysql"), nullable=True)
        )
        batch_op.create_index(
            "index_issues_experiment_last_seen", ["experiment_id", "last_seen_timestamp"]
        )
        batch_op.create_index(
            "index_issues_experiment_updated", ["experiment_id", "last_updated_timestamp"]
        )
    op.create_table(
        "issue_categories",
        sa.Column("issue_id", sa.String(36), nullable=False),
        sa.Column("category_hash", sa.String(64), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("issue_id", "category_hash", name="issue_categories_pk"),
        sa.ForeignKeyConstraint(
            ["issue_id"],
            ["issues.issue_id"],
            name="fk_issue_categories_issue_id",
            ondelete="CASCADE",
        ),
    )
    op.create_index("index_issue_categories_hash", "issue_categories", ["category_hash"])
    issues = sa.table(
        "issues",
        sa.column("issue_id"),
        sa.column("categories"),
        sa.column("created_timestamp"),
        sa.column("last_seen_timestamp"),
        sa.column("source_run_id"),
    )
    conn = op.get_bind()
    conn.execute(issues.update().values(last_seen_timestamp=issues.c.created_timestamp))
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
                "category_hash": category_hash,
            }
            for row in batch
            for category_hash, category in {
                hashlib.sha256(category.strip().lower().encode()).hexdigest(): category
                for category in json.loads(row.categories)
            }.items()
        ]
        if entries:
            conn.execute(categories.insert(), entries)


def downgrade():
    op.drop_index("index_issue_categories_hash", table_name="issue_categories")
    op.drop_table("issue_categories")
    with op.batch_alter_table("issues") as batch_op:
        batch_op.drop_index("index_issues_experiment_updated")
        batch_op.drop_index("index_issues_experiment_last_seen")
        batch_op.drop_column("metadata")
        batch_op.drop_column("last_updated_by")
        batch_op.drop_column("last_seen_timestamp")
