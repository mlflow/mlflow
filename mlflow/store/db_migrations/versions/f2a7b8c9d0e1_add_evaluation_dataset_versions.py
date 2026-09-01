"""add immutable evaluation dataset versions

Revision ID: f2a7b8c9d0e1
Revises: b7e2c1a4d9f3
Create Date: 2026-08-25 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

revision = "f2a7b8c9d0e1"
down_revision = "b7e2c1a4d9f3"
branch_labels = None
depends_on = None


def _get_json_type():
    return mssql.JSON if op.get_bind().dialect.name == "mssql" else sa.JSON


def upgrade():
    json_type = _get_json_type()
    with op.batch_alter_table("evaluation_datasets") as batch_op:
        batch_op.add_column(
            sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1"))
        )

    op.create_table(
        "evaluation_dataset_versions",
        sa.Column("dataset_id", sa.String(36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("schema", sa.Text(), nullable=True),
        sa.Column("profile", sa.Text(), nullable=True),
        sa.Column("digest", sa.String(64), nullable=True),
        sa.Column("created_time", sa.BigInteger(), nullable=False),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("operation", sa.String(32), nullable=True),
        sa.PrimaryKeyConstraint("dataset_id", "version", name="evaluation_dataset_versions_pk"),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["evaluation_datasets.dataset_id"],
            name="fk_evaluation_dataset_versions_dataset_id", ondelete="CASCADE"
        ),
    )
    op.create_index(
        "index_evaluation_dataset_versions_dataset_id",
        "evaluation_dataset_versions", ["dataset_id"], unique=False
    )

    op.create_table(
        "evaluation_dataset_version_records",
        sa.Column("dataset_id", sa.String(36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("dataset_record_id", sa.String(36), nullable=False),
        sa.Column("inputs", json_type, nullable=False),
        sa.Column("outputs", json_type, nullable=True),
        sa.Column("expectations", json_type, nullable=True),
        sa.Column("tags", json_type, nullable=True),
        sa.Column("source", json_type, nullable=True),
        sa.Column("source_id", sa.String(36), nullable=True),
        sa.Column("source_type", sa.String(255), nullable=True),
        sa.Column("created_time", sa.BigInteger(), nullable=True),
        sa.Column("last_update_time", sa.BigInteger(), nullable=True),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("last_updated_by", sa.String(255), nullable=True),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.PrimaryKeyConstraint(
            "dataset_id", "version", "dataset_record_id",
            name="evaluation_dataset_version_records_pk"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id", "version"],
            ["evaluation_dataset_versions.dataset_id", "evaluation_dataset_versions.version"],
            name="fk_evaluation_dataset_version_records_version", ondelete="CASCADE"
        ),
    )
    op.create_index(
        "index_evaluation_dataset_version_records_dataset_version",
        "evaluation_dataset_version_records", ["dataset_id", "version"], unique=False
    )

    bind = op.get_bind()
    rows = bind.execute(sa.text(
        "SELECT dataset_id, schema, profile, digest, last_update_time, last_updated_by "
        "FROM evaluation_datasets"
    )).mappings().all()
    if rows:
        versions = sa.table(
            "evaluation_dataset_versions",
            sa.column("dataset_id"), sa.column("version"), sa.column("schema"),
            sa.column("profile"), sa.column("digest"), sa.column("created_time"),
            sa.column("created_by"), sa.column("operation"),
        )
        op.bulk_insert(versions, [
            {
                "dataset_id": r["dataset_id"], "version": 1, "schema": r["schema"],
                "profile": r["profile"], "digest": r["digest"],
                "created_time": r["last_update_time"] or 0,
                "created_by": r["last_updated_by"], "operation": "baseline",
            }
            for r in rows
        ])


def downgrade():
    op.drop_table("evaluation_dataset_version_records")
    op.drop_table("evaluation_dataset_versions")
    with op.batch_alter_table("evaluation_datasets") as batch_op:
        batch_op.drop_column("version")
