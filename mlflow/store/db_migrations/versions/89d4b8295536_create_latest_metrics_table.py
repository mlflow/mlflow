"""create latest metrics table

Revision ID: 89d4b8295536
Revises: 7ac759974ad8
Create Date: 2019-08-20 11:53:28.178479

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import orm, func, and_
from sqlalchemy import (
    Column, String, ForeignKey, Integer, Float, 
    BigInteger, PrimaryKeyConstraint, Boolean)
from mlflow.store.dbmodels.models import SqlRun, SqlMetric, SqlLatestMetric 


# revision identifiers, used by Alembic.
revision = '89d4b8295536'
down_revision = '7ac759974ad8'
branch_labels = None
depends_on = None


def get_latest_metrics_for_run(session, run_uuid):
    metrics_with_max_step = session \
        .query(SqlMetric.run_uuid, SqlMetric.key, func.max(SqlMetric.step).label('step')) \
        .filter(SqlMetric.run_uuid == run_uuid) \
        .group_by(SqlMetric.key, SqlMetric.run_uuid) \
        .subquery('metrics_with_max_step')
    metrics_with_max_timestamp = session \
        .query(SqlMetric.run_uuid, SqlMetric.key, SqlMetric.step,
               func.max(SqlMetric.timestamp).label('timestamp')) \
        .filter(SqlMetric.run_uuid == run_uuid) \
        .join(metrics_with_max_step,
              and_(SqlMetric.step == metrics_with_max_step.c.step,
                   SqlMetric.run_uuid == metrics_with_max_step.c.run_uuid,
                   SqlMetric.key == metrics_with_max_step.c.key)) \
        .group_by(SqlMetric.key, SqlMetric.run_uuid, SqlMetric.step) \
        .subquery('metrics_with_max_timestamp')
    metrics_with_max_value = session \
        .query(SqlMetric.run_uuid, SqlMetric.key, SqlMetric.step, SqlMetric.timestamp,
               func.max(SqlMetric.value).label('value'), SqlMetric.is_nan) \
        .filter(SqlMetric.run_uuid == run_uuid) \
        .join(metrics_with_max_timestamp,
              and_(SqlMetric.timestamp == metrics_with_max_timestamp.c.timestamp,
                   SqlMetric.run_uuid == metrics_with_max_timestamp.c.run_uuid,
                   SqlMetric.key == metrics_with_max_timestamp.c.key,
                   SqlMetric.step == metrics_with_max_timestamp.c.step)) \
        .group_by(SqlMetric.run_uuid, SqlMetric.key,
                  SqlMetric.step, SqlMetric.timestamp, SqlMetric.is_nan) \
        .all()
    return metrics_with_max_value


def upgrade():
    op.create_table(SqlLatestMetric.__tablename__,
        Column('key', String(length=250)),
        Column('value', Float(precision=53), nullable=False),
        Column('timestamp', BigInteger, nullable=False),
        Column('step', BigInteger, default=0, nullable=False),
        Column('is_nan', Boolean, default=False, nullable=False),
        Column('run_uuid', String(length=32), ForeignKey('runs.run_uuid'), 
                  primary_key=True, nullable=False),
        PrimaryKeyConstraint('key', 'run_uuid', name='latest_metric_pk')
    )

    bind = op.get_bind()
    session = orm.Session(bind=bind)
    all_run_uuids = session.query(SqlRun.run_uuid).all()
    for run_uuid in all_run_uuids:
        run_latest_metrics = get_latest_metrics_for_run
        print(run_latest_metrics)


def downgrade():
    pass
