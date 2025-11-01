"""add secrets tables

Create Date: 2025-01-27 00:00:00.000000

"""

import time

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "1b49d398cd23"
down_revision = "bf29a5ff90ea"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "secrets",
        sa.Column("secret_id", sa.String(length=36), nullable=False),
        sa.Column("secret_name", sa.String(length=255), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("iv", sa.LargeBinary(), nullable=False),
        sa.Column("wrapped_dek", sa.LargeBinary(), nullable=False),
        sa.Column("kek_version", sa.Integer(), nullable=False),
        sa.Column("aad_hash", sa.LargeBinary(), nullable=False),
        sa.Column("is_shared", sa.Boolean(), nullable=False, default=False),
        sa.Column("state", sa.String(length=36), nullable=False, default="ACTIVE"),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column("last_updated_by", sa.String(length=255), nullable=True),
        sa.Column(
            "last_updated_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("secret_id", name="secrets_pk"),
        sa.UniqueConstraint("secret_name", name="unique_secret_name"),
    )
    with op.batch_alter_table("secrets", schema=None) as batch_op:
        batch_op.create_index(
            "index_secrets_is_shared_secret_name", ["is_shared", "secret_name"], unique=False
        )
        batch_op.create_index("index_secrets_state", ["state"], unique=False)

    op.create_table(
        "secrets_bindings",
        sa.Column("binding_id", sa.String(length=36), nullable=False),
        sa.Column("secret_id", sa.String(length=36), nullable=False),
        sa.Column("resource_type", sa.String(length=50), nullable=False),
        sa.Column("resource_id", sa.String(length=255), nullable=False),
        sa.Column("field_name", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "last_updated_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column("last_updated_by", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["secret_id"],
            ["secrets.secret_id"],
            name="fk_secrets_bindings_secret_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("binding_id", name="secrets_bindings_pk"),
        sa.UniqueConstraint(
            "resource_type", "resource_id", "field_name", name="unique_binding_per_resource"
        ),
    )
    with op.batch_alter_table("secrets_bindings", schema=None) as batch_op:
        batch_op.create_index("index_secrets_bindings_secret_id", ["secret_id"], unique=False)
        batch_op.create_index(
            "index_secrets_bindings_resource_type_resource_id",
            ["resource_type", "resource_id"],
            unique=False,
        )


def downgrade():
    op.drop_table("secrets_bindings")
    op.drop_table("secrets")
