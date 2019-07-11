"""allow nulls for metric values

Revision ID: 181f10493468
Revises: 90e64c465722
Create Date: 2019-07-10 22:40:18.787993

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import orm, Column, Integer, String, ForeignKey, PrimaryKeyConstraint

# revision identifiers, used by Alembic.
revision = '181f10493468'
down_revision = '90e64c465722'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('metrics', sa.Column('is_nan', sa.Boolean(), nullable=False, server_default='0'))
    # bind = op.get_bind()
    # session = orm.Session(bind=bind)
    # metrics = session.query(SqlMetric).all()
    # for metric in metrics:
    #     metric.is_nan = False
    #     session.merge(metric)
    # session.commit()
    with op.batch_alter_table("metrics") as batch_op:
        batch_op.drop_constraint(constraint_name='metric_pk', type_="primary")
        batch_op.create_primary_key(
            constraint_name='metric_pk',
            columns=['key', 'timestamp', 'step', 'run_uuid', 'value', 'is_nan'])


def downgrade():
    pass
