"""add scorer preset tables

Create Date: 2026-08-06 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

from mlflow.store.tracking.dbmodels.models import (
    SqlScorerPreset,
    SqlScorerPresetMembership,
    SqlScorerPresetVersion,
)

# revision identifiers, used by Alembic.
revision = "0e7f11fcd8ac"
down_revision = "6f8d9c3b2a1e"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        SqlScorerPreset.__tablename__,
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("preset_name", sa.String(length=256), nullable=False),
        sa.Column("preset_id", sa.String(length=36), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.experiment_id"],
            name="fk_scorer_presets_experiment_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("preset_id", name="scorer_preset_pk"),
    )

    op.create_table(
        SqlScorerPresetVersion.__tablename__,
        sa.Column("preset_id", sa.String(length=36), nullable=False),
        sa.Column("version_hash", sa.String(length=64), nullable=False),
        sa.Column("creation_time", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["preset_id"],
            ["scorer_presets.preset_id"],
            name="fk_scorer_preset_versions_preset_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("preset_id", "version_hash", name="scorer_preset_version_pk"),
    )

    op.create_table(
        SqlScorerPresetMembership.__tablename__,
        sa.Column("preset_id", sa.String(length=36), nullable=False),
        sa.Column("version_hash", sa.String(length=64), nullable=False),
        sa.Column("scorer_id", sa.String(length=36), nullable=False),
        sa.Column("scorer_version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["preset_id", "version_hash"],
            [
                "scorer_preset_versions.preset_id",
                "scorer_preset_versions.version_hash",
            ],
            name="fk_preset_membership_version",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "preset_id",
            "version_hash",
            "scorer_id",
            name="scorer_preset_membership_pk",
        ),
    )

    with op.batch_alter_table(SqlScorerPreset.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlScorerPreset.__tablename__}_experiment_id_preset_name",
            ["experiment_id", "preset_name"],
            unique=True,
        )

    with op.batch_alter_table(SqlScorerPresetVersion.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlScorerPresetVersion.__tablename__}_preset_id",
            ["preset_id"],
            unique=False,
        )

    with op.batch_alter_table(SqlScorerPresetMembership.__tablename__, schema=None) as batch_op:
        batch_op.create_index(
            f"index_{SqlScorerPresetMembership.__tablename__}_scorer_id",
            ["scorer_id"],
            unique=False,
        )


def downgrade():
    op.drop_table(SqlScorerPresetMembership.__tablename__)
    op.drop_table(SqlScorerPresetVersion.__tablename__)
    op.drop_table(SqlScorerPreset.__tablename__)
