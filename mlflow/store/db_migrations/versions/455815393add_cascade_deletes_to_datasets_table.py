"""cascade deletes to datasets table

Revision ID: 455815393add
Revises: acf3f17fdcc7
Create Date: 2023-12-18 10:29:31.278723

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "455815393add"
down_revision = "acf3f17fdcc7"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()

    # SQLite operates slightly differently from other databases as it doesn't
    # provide constraint names for foreign keys. We need to provide this
    # naming_convention argument to op.batch_alter_table in order to drop
    # and create a new foreign key constraint.
    #
    # See the following link for more details:
    # https://alembic.sqlalchemy.org/en/latest/batch.html#dropping-unnamed-or-named-foreign-key-constraints
    naming_convention = {
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    }
    new_constraint_name = "fk_datasets_experiment_id_experiments"

    constraints = sa.inspect(bind).get_foreign_keys("datasets")
    old_constraint_name = (
        [c["name"] for c in constraints if c["constrained_columns"][0] == "experiment_id"][0]
        if bind.engine.name != "sqlite"
        else new_constraint_name
    )

    with op.batch_alter_table("datasets", naming_convention=naming_convention) as batch_op:
        batch_op.drop_constraint(old_constraint_name, type_="foreignkey")
        batch_op.create_foreign_key(
            new_constraint_name,
            "experiments",
            ["experiment_id"],
            ["experiment_id"],
            ondelete="CASCADE",
        )


def downgrade():
    pass
