"""add skill registry tables

Create Date: 2026-09-01

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql

revision = "e7d1f4b2a9c6"
down_revision = "b7e2c1a4d9f3"
branch_labels = None
depends_on = None


def _get_json_type():
    dialect_name = op.get_bind().dialect.name
    if dialect_name == "mssql":
        return mssql.JSON
    else:
        return sa.JSON


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
        sa.Column("description", sa.String(length=5000), nullable=True),
        sa.Column("icons", json_type, nullable=True),
        sa.Column("search_text", sa.Text(), nullable=True),
        sa.Column("imported_keywords_json", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("last_updated_by", sa.String(length=256), nullable=True),
        sa.Column("creation_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_timestamp", sa.BigInteger(), nullable=False),
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
        sa.Column("creation_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_timestamp", sa.BigInteger(), nullable=False),
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
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
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
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
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

    op.create_table(
        "agent_plugins",
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
        sa.Column("description", sa.String(length=5000), nullable=True),
        sa.Column("icons", json_type, nullable=True),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("last_updated_by", sa.String(length=256), nullable=True),
        sa.Column("creation_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_timestamp", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("workspace", "organization", "name", name="agent_plugins_pk"),
    )

    op.create_table(
        "agent_plugin_versions",
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
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("version_major", sa.Integer(), nullable=False),
        sa.Column("version_minor", sa.Integer(), nullable=False),
        sa.Column("version_patch", sa.Integer(), nullable=False),
        sa.Column("version_prerelease_sort_key", sa.String(length=512), nullable=False),
        sa.Column("plugin_json", json_type, nullable=False),
        sa.Column("search_text", sa.Text(), nullable=True),
        sa.Column("source_type", sa.String(length=20), nullable=True),
        sa.Column("source", sa.String(length=2048), nullable=True),
        sa.Column("ref", sa.String(length=2048), nullable=True),
        sa.Column("subpath", sa.String(length=2048), nullable=True),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("last_updated_by", sa.String(length=256), nullable=True),
        sa.Column("creation_timestamp", sa.BigInteger(), nullable=False),
        sa.Column("last_updated_timestamp", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace", "organization", "name"],
            ["agent_plugins.workspace", "agent_plugins.organization", "agent_plugins.name"],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="agent_plugin_versions_plugin_fkey",
        ),
        sa.PrimaryKeyConstraint(
            "workspace", "organization", "name", "version", name="agent_plugin_versions_pk"
        ),
    )

    op.create_table(
        "agent_plugin_tags",
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
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["workspace", "organization", "name"],
            ["agent_plugins.workspace", "agent_plugins.organization", "agent_plugins.name"],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="agent_plugin_tags_plugin_fkey",
        ),
        sa.PrimaryKeyConstraint(
            "workspace", "organization", "name", "key", name="agent_plugin_tags_pk"
        ),
    )

    op.create_table(
        "agent_plugin_version_tags",
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
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["workspace", "organization", "name", "version"],
            [
                "agent_plugin_versions.workspace",
                "agent_plugin_versions.organization",
                "agent_plugin_versions.name",
                "agent_plugin_versions.version",
            ],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="agent_plugin_version_tags_version_fkey",
        ),
        sa.PrimaryKeyConstraint(
            "workspace",
            "organization",
            "name",
            "version",
            "key",
            name="agent_plugin_version_tags_pk",
        ),
    )

    op.create_table(
        "agent_plugin_aliases",
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
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace", "organization", "name"],
            ["agent_plugins.workspace", "agent_plugins.organization", "agent_plugins.name"],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="agent_plugin_aliases_plugin_fkey",
        ),
        sa.PrimaryKeyConstraint(
            "workspace", "organization", "name", "alias", name="agent_plugin_aliases_pk"
        ),
    )

    op.create_table(
        "agent_plugin_version_members",
        sa.Column(
            "plugin_workspace",
            sa.String(length=63),
            nullable=False,
            server_default=sa.text("'default'"),
        ),
        sa.Column(
            "plugin_organization",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("plugin_name", sa.String(length=128), nullable=False),
        sa.Column("plugin_version", sa.String(length=128), nullable=False),
        sa.Column("member_name", sa.String(length=128), nullable=False),
        sa.Column(
            "member_organization",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("member_version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["plugin_workspace", "plugin_organization", "plugin_name", "plugin_version"],
            [
                "agent_plugin_versions.workspace",
                "agent_plugin_versions.organization",
                "agent_plugin_versions.name",
                "agent_plugin_versions.version",
            ],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="agent_plugin_version_members_plugin_fkey",
        ),
        # NO ACTION (not RESTRICT): SQL Server has no RESTRICT keyword, so it
        # would break the MSSQL migration; NO ACTION still blocks the delete on
        # every dialect and is the MLflow-wide convention. Keep in sync with
        # SqlAgentPluginVersionMember in dbmodels/models.py.
        sa.ForeignKeyConstraint(
            ["plugin_workspace", "member_organization", "member_name", "member_version"],
            [
                "skill_versions.workspace",
                "skill_versions.organization",
                "skill_versions.name",
                "skill_versions.version",
            ],
            ondelete="NO ACTION",
            name="agent_plugin_version_members_skill_fkey",
        ),
        sa.PrimaryKeyConstraint(
            "plugin_workspace",
            "plugin_organization",
            "plugin_name",
            "plugin_version",
            "member_name",
            name="agent_plugin_version_members_pk",
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
    # Keep this index narrow enough for MySQL's 3072-byte key limit: the coarse
    # SemVer core prefix prunes candidates, and the prerelease sort key (excluded
    # here) refines ordering at query time.
    op.create_index(
        "ix_agent_plugin_versions_latest_lookup",
        "agent_plugin_versions",
        [
            "workspace",
            "organization",
            "name",
            "status",
            "version_major",
            "version_minor",
            "version_patch",
            "creation_timestamp",
        ],
    )
    op.create_index(
        "ix_agent_plugin_version_members_skill_fkey",
        "agent_plugin_version_members",
        ["plugin_workspace", "member_organization", "member_name", "member_version"],
    )


def downgrade():
    op.drop_index(
        "ix_agent_plugin_version_members_skill_fkey",
        table_name="agent_plugin_version_members",
    )
    op.drop_index("ix_agent_plugin_versions_latest_lookup", table_name="agent_plugin_versions")
    op.drop_index("ix_skill_versions_digest", table_name="skill_versions")
    op.drop_index("ix_skill_versions_latest_lookup", table_name="skill_versions")

    op.drop_table("agent_plugin_version_members")
    op.drop_table("agent_plugin_aliases")
    op.drop_table("agent_plugin_version_tags")
    op.drop_table("agent_plugin_tags")
    op.drop_table("agent_plugin_versions")
    op.drop_table("agent_plugins")
    op.drop_table("skill_aliases")
    op.drop_table("skill_version_tags")
    op.drop_table("skill_tags")
    op.drop_table("skill_versions")
    op.drop_table("skills")
