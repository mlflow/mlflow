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
    dialect = context.dialect.name
    substring = sa.func.substr if dialect == "sqlite" else sa.func.substring
    path = substring(source, len("models:") + 1, 500)
    model_id = sa.func.replace(path, "/", "")

    # Tracking and registry databases can be separate. Recover only a direct logged-model URI.
    if dialect == "mssql":
        binary_path = path.collate("Latin1_General_100_BIN2")
        valid_path = binary_path.not_like("%[^-/a-zA-Z0-9_]%") & binary_path.not_like("%[^/]/%")
    elif dialect == "sqlite":
        # GLOB works in offline SQL, and SQLite's string functions stop at a NUL byte.
        valid_path = (
            ~path.bool_op("GLOB")("*[^/a-zA-Z0-9_-]*")
            & ~path.bool_op("GLOB")("*[^/]/*")
            & (sa.func.instr(source, sa.func.char(0)) == 0)
        )
    else:
        valid_path = ~path.regexp_match("[^/a-zA-Z0-9_-]|[^/]/")
    op.execute(
        versions
        .update()
        .where(
            versions.c.model_id.is_(None),
            sa.func.lower(substring(source, 1, 8)) == "models:/",
            # Two leading slashes introduce an authority. Three or more introduce an empty one.
            ~path.like("//%") | path.like("///%"),
            sa.func.length(model_id).between(1, 36),
            valid_path,
        )
        .values(model_id=model_id)
    )


def downgrade():
    context = op.get_context()
    if (
        context.dialect.name == "sqlite"
        and not context.as_sql
        and context.dialect.server_version_info < (3, 35)
    ):
        with op.batch_alter_table("model_versions") as batch_op:
            batch_op.drop_column("model_id")
    else:
        op.drop_column("model_versions", "model_id")
