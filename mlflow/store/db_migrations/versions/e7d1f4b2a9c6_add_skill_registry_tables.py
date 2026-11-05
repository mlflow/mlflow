"""add skill registry tables

Create Date: 2026-09-01

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql, mysql

revision = "e7d1f4b2a9c6"
down_revision = "dc11669786a5"
branch_labels = None
depends_on = None


def _get_json_type():
    dialect_name = op.get_bind().dialect.name
    if dialect_name == "mssql":
        return mssql.JSON
    else:
        return sa.JSON


SKILL_REGISTRY_TAG_KEY_STRING = (
    sa
    .String(length=250)
    .with_variant(mysql.VARCHAR(250, collation="utf8mb4_bin"), "mysql")
    .with_variant(mssql.VARCHAR(250, collation="SQL_Latin1_General_CP1_CS_AS"), "mssql")
)

SKILL_REGISTRY_TAG_VALUE_TEXT = (
    sa.Text().with_variant(mysql.MEDIUMTEXT, "mysql").with_variant(mssql.NVARCHAR(None), "mssql")
)


def upgrade():
    json_type = _get_json_type()

    op.create_table(
        "skills",
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column(
            "organization",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("icons", json_type, nullable=True),
        sa.Column("search_text", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("last_updated_by", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_at", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("workspace", "organization", "name", name="skills_pk"),
    )

    op.create_table(
        "skill_versions",
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column(
            "organization",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.String(length=20), nullable=True),
        sa.Column("source", sa.String(length=2048), nullable=True),
        sa.Column("ref", sa.String(length=2048), nullable=True),
        sa.Column("subpath", sa.String(length=2048), nullable=True),
        sa.Column("digest", sa.String(length=64), nullable=True),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("last_updated_by", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_at", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace", "organization", "name"],
            ["skills.workspace", "skills.organization", "skills.name"],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="skill_versions_skill_fkey",
        ),
        sa.PrimaryKeyConstraint(
            "workspace", "organization", "name", "version", name="skill_versions_pk"
        ),
    )

    op.create_table(
        "skill_tags",
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column(
            "organization",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("key", SKILL_REGISTRY_TAG_KEY_STRING, nullable=False),
        sa.Column("value", SKILL_REGISTRY_TAG_VALUE_TEXT, nullable=True),
        sa.ForeignKeyConstraint(
            ["workspace", "organization", "name"],
            ["skills.workspace", "skills.organization", "skills.name"],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="skill_tags_skill_fkey",
        ),
        sa.PrimaryKeyConstraint("workspace", "organization", "name", "key", name="skill_tags_pk"),
    )

    op.create_table(
        "skill_version_tags",
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column(
            "organization",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("key", SKILL_REGISTRY_TAG_KEY_STRING, nullable=False),
        sa.Column("value", SKILL_REGISTRY_TAG_VALUE_TEXT, nullable=True),
        sa.ForeignKeyConstraint(
            ["workspace", "organization", "name", "version"],
            [
                "skill_versions.workspace",
                "skill_versions.organization",
                "skill_versions.name",
                "skill_versions.version",
            ],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="skill_version_tags_version_fkey",
        ),
        sa.PrimaryKeyConstraint(
            "workspace",
            "organization",
            "name",
            "version",
            "key",
            name="skill_version_tags_pk",
        ),
    )

    op.create_table(
        "skill_aliases",
        sa.Column(
            "workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column(
            "organization",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("alias", sa.String(length=256), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace", "organization", "name"],
            ["skills.workspace", "skills.organization", "skills.name"],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="skill_aliases_skill_fkey",
        ),
        sa.PrimaryKeyConstraint(
            "workspace", "organization", "name", "alias", name="skill_aliases_pk"
        ),
    )

    op.create_index(
        "ix_skill_versions_latest_lookup",
        "skill_versions",
        ["workspace", "organization", "name", "status", "version"],
    )
    op.create_index(
        "ix_skill_versions_digest",
        "skill_versions",
        ["workspace", "organization", "name", "digest"],
    )


def downgrade():
    # Drop the tables in FK-safe order (children before parents). DROP TABLE removes
    # each table's indexes along with it, so they need no separate handling.
    op.drop_table("skill_aliases")
    op.drop_table("skill_version_tags")
    op.drop_table("skill_tags")
    op.drop_table("skill_versions")
    op.drop_table("skills")
