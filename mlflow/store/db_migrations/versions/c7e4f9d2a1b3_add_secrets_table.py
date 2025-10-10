"""add secrets table

Create Date: 2025-10-08 18:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c7e4f9d2a1b3"
down_revision = "bf29a5ff90ea"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "secrets",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("scope", sa.Integer(), nullable=False),
        sa.Column("scope_id", sa.Integer(), nullable=True),
        sa.Column("name_hash", sa.String(length=64), nullable=False),
        sa.Column("encrypted_name", sa.Text(), nullable=False),
        sa.Column("secret", sa.Text(), nullable=False),
        sa.Column("encrypted_dek", sa.Text(), nullable=True),
        sa.Column("master_key_version", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scope", "scope_id", "name_hash", name="uq_secret_scope_name"),
    )
    with op.batch_alter_table("secrets", schema=None) as batch_op:
        batch_op.create_index("idx_secrets_scope", ["scope", "scope_id"], unique=False)


def downgrade():
    op.drop_table("secrets")
