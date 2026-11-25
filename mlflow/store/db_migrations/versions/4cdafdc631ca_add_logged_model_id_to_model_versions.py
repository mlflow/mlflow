"""Persist the logged model ID on SQL model versions."""

import sqlalchemy as sa
from alembic import op

revision = "4cdafdc631ca"
down_revision = "f4a9b2c7d1e6"
branch_labels = None
depends_on = None


def upgrade():
    context = op.get_context()
    # MySQL and SQLite can retain the column if the backfill is interrupted.
    if context.as_sql or not any(
        column["name"] == "model_id"
        for column in sa.inspect(op.get_bind()).get_columns("model_versions")
    ):
        op.add_column("model_versions", sa.Column("model_id", sa.String(36), nullable=True))
    versions = sa.table("model_versions", sa.column("source", sa.String()), sa.column("model_id"))
    source = versions.c.source
    prefix = "models:/"
    dialect = context.dialect.name
    substring = sa.func.substr if dialect == "sqlite" else sa.func.substring
    model_id = substring(source, len(prefix) + 1, 500)

    # Tracking and registry databases can be separate. Recover only a direct logged-model URI.
    if dialect == "mssql":
        valid_id = model_id.not_like("%[^a-zA-Z0-9_-]%")
    else:
        valid_id = ~model_id.regexp_match("[^a-zA-Z0-9_-]")
    op.execute(
        versions
        .update()
        .where(
            versions.c.model_id.is_(None),
            source.like(f"{prefix}_%"),
            sa.func.length(model_id).between(1, 36),
            valid_id,
        )
        .values(model_id=model_id)
    )


def downgrade():
    op.drop_column("model_versions", "model_id")
