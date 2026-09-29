"""backfill skill search text and tag value type

Create Date: 2026-09-29

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = "4b75f9a0c123"
down_revision = "e7d1f4b2a9c6"
branch_labels = None
depends_on = None


def _skill_search_text_expression(dialect_name: str) -> str:
    if dialect_name == "mysql":
        return (
            "CASE WHEN description IS NULL OR description = '' THEN name "
            "ELSE CONCAT(name, ' ', description) END"
        )
    if dialect_name == "mssql":
        return (
            "CASE WHEN description IS NULL OR description = '' THEN name "
            "ELSE name + ' ' + description END"
        )
    return (
        "CASE WHEN description IS NULL OR description = '' THEN name "
        "ELSE name || ' ' || description END"
    )


def _backfill_skill_search_text(dialect_name: str) -> None:
    op.execute(
        sa.text(
            f"""
            UPDATE skills
            SET search_text = {_skill_search_text_expression(dialect_name)}
            WHERE search_text IS NULL
            """
        )
    )


def _upgrade_mysql_tag_value_columns() -> None:
    for table_name in (
        "skill_tags",
        "skill_version_tags",
        "agent_plugin_tags",
        "agent_plugin_version_tags",
    ):
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.alter_column(
                "value",
                existing_type=sa.Text(),
                type_=mysql.MEDIUMTEXT,
                existing_nullable=True,
            )


def upgrade():
    dialect_name = op.get_bind().dialect.name
    _backfill_skill_search_text(dialect_name)
    if dialect_name == "mysql":
        _upgrade_mysql_tag_value_columns()


def downgrade():
    pass
