"""add semver-derived MCP server version columns

Revision ID: d9f8e7c6b5a4
Revises: a8b9c0d1e2f3

Create Date: 2026-06-24 12:30:00.000000

"""

import re

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "d9f8e7c6b5a4"
down_revision = "a8b9c0d1e2f3"
branch_labels = None
depends_on = None

_MAX_SEMVER_CORE_VALUE = 2_147_483_647
_RELEASE_PRERELEASE_SORT_KEY = "2"
_ASCII_CODE_WIDTH = 3
_TEXT_IDENTIFIER_TERMINATOR = "000"
_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


def _encode_numeric_prerelease_identifier(identifier: str) -> str:
    return f"0{len(identifier):03d}{identifier}"


def _encode_ascii_sort_key(value: str) -> str:
    return "".join(f"{ord(char):0{_ASCII_CODE_WIDTH}d}" for char in value)


def _encode_text_prerelease_identifier(identifier: str) -> str:
    return f"1{_encode_ascii_sort_key(identifier)}{_TEXT_IDENTIFIER_TERMINATOR}"


def _parse_and_encode_semver(version: str) -> tuple[int, int, int, str]:
    match = _SEMVER_RE.fullmatch(version)
    if not match:
        raise RuntimeError(
            "Cannot migrate MCP server versions because existing version "
            f"{version!r} is not valid SemVer."
        )

    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))
    for label, value in (("major", major), ("minor", minor), ("patch", patch)):
        if value > _MAX_SEMVER_CORE_VALUE:
            raise RuntimeError(
                "Cannot migrate MCP server versions because existing version "
                f"{version!r} has {label} component {value} which exceeds the "
                f"supported maximum of {_MAX_SEMVER_CORE_VALUE}."
            )

    prerelease = tuple(match.group("prerelease").split(".")) if match.group("prerelease") else ()
    if not prerelease:
        prerelease_sort_key = _RELEASE_PRERELEASE_SORT_KEY
    else:
        parts = []
        for identifier in prerelease:
            if identifier.isdigit():
                parts.append(_encode_numeric_prerelease_identifier(identifier))
            else:
                parts.append(_encode_text_prerelease_identifier(identifier))
        prerelease_sort_key = "".join(parts)
    return major, minor, patch, prerelease_sort_key


def upgrade():
    op.add_column(
        "mcp_server_versions",
        sa.Column("version_major", sa.Integer(), nullable=True),
    )
    op.add_column(
        "mcp_server_versions",
        sa.Column("version_minor", sa.Integer(), nullable=True),
    )
    op.add_column(
        "mcp_server_versions",
        sa.Column("version_patch", sa.Integer(), nullable=True),
    )
    op.add_column(
        "mcp_server_versions",
        sa.Column("version_prerelease_sort_key", sa.String(length=512), nullable=True),
    )

    bind = op.get_bind()
    rows = bind.execute(
        sa.text("SELECT workspace, name, version FROM mcp_server_versions")
    ).fetchall()
    for row in rows:
        major, minor, patch, prerelease_sort_key = _parse_and_encode_semver(row.version)
        bind.execute(
            sa.text(
                """
                UPDATE mcp_server_versions
                SET version_major = :version_major,
                    version_minor = :version_minor,
                    version_patch = :version_patch,
                    version_prerelease_sort_key = :version_prerelease_sort_key
                WHERE workspace = :workspace AND name = :name AND version = :version
                """
            ),
            {
                "workspace": row.workspace,
                "name": row.name,
                "version": row.version,
                "version_major": major,
                "version_minor": minor,
                "version_patch": patch,
                "version_prerelease_sort_key": prerelease_sort_key,
            },
        )

    with op.batch_alter_table("mcp_server_versions") as batch_op:
        batch_op.alter_column(
            "version_major",
            existing_type=sa.Integer(),
            nullable=False,
        )
        batch_op.alter_column(
            "version_minor",
            existing_type=sa.Integer(),
            nullable=False,
        )
        batch_op.alter_column(
            "version_patch",
            existing_type=sa.Integer(),
            nullable=False,
        )

    op.drop_index("idx_mcp_server_versions_latest", table_name="mcp_server_versions")
    op.create_index(
        "idx_mcp_server_versions_latest",
        "mcp_server_versions",
        [
            "workspace",
            "name",
            "status",
            sa.text("version_major DESC"),
            sa.text("version_minor DESC"),
            sa.text("version_patch DESC"),
        ],
    )
    with op.batch_alter_table("mcp_servers") as batch_op:
        batch_op.drop_column("latest_version")


def downgrade():
    with op.batch_alter_table("mcp_servers") as batch_op:
        batch_op.add_column(sa.Column("latest_version", sa.String(length=128), nullable=True))
    op.drop_index("idx_mcp_server_versions_latest", table_name="mcp_server_versions")
    op.create_index(
        "idx_mcp_server_versions_latest",
        "mcp_server_versions",
        ["workspace", "name", "status", sa.text("created_at DESC"), sa.text("version DESC")],
    )
    op.drop_column("mcp_server_versions", "version_prerelease_sort_key")
    op.drop_column("mcp_server_versions", "version_patch")
    op.drop_column("mcp_server_versions", "version_minor")
    op.drop_column("mcp_server_versions", "version_major")
