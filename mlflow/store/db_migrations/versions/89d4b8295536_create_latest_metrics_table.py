"""create latest metrics table

Revision ID: 89d4b8295536
Revises: 7ac759974ad8
Create Date: 2019-08-20 11:53:28.178479

"""
from alembic import op
from sqlalchemy import orm, func, and_
from sqlalchemy import (
    Column, String, ForeignKey, Float,
    BigInteger, PrimaryKeyConstraint, Boolean)
from mlflow.store.dbmodels.models import SqlMetric, SqlLatestMetric


# revision identifiers, used by Alembic.
revision = '89d4b8295536'
down_revision = '7ac759974ad8'
branch_labels = None
depends_on = None


def _get_latest_metrics_for_runs(session):
    metrics_with_max_step = session \
        .query(SqlMetric.run_uuid, SqlMetric.key, func.max(SqlMetric.step).label('step')) \
        .group_by(SqlMetric.key, SqlMetric.run_uuid) \
        .subquery('metrics_with_max_step')
    metrics_with_max_timestamp = session \
        .query(SqlMetric.run_uuid, SqlMetric.key, SqlMetric.step,
               func.max(SqlMetric.timestamp).label('timestamp')) \
        .join(metrics_with_max_step,
              and_(SqlMetric.step == metrics_with_max_step.c.step,
                   SqlMetric.run_uuid == metrics_with_max_step.c.run_uuid,
                   SqlMetric.key == metrics_with_max_step.c.key)) \
        .group_by(SqlMetric.key, SqlMetric.run_uuid, SqlMetric.step) \
        .subquery('metrics_with_max_timestamp')
    metrics_with_max_value = session \
        .query(SqlMetric.run_uuid, SqlMetric.key, SqlMetric.step, SqlMetric.timestamp,
               func.max(SqlMetric.value).label('value'), SqlMetric.is_nan) \
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
        Column('run_uuid', String(length=32), ForeignKey('runs.run_uuid'), nullable=False),
        PrimaryKeyConstraint('key', 'run_uuid', name='latest_metric_pk')
    )

    bind = op.get_bind()
    session = orm.Session(bind=bind)

    from datetime import datetime
    begin = datetime.now()
    all_latest_metrics = _get_latest_metrics_for_runs(session=session)
    session.add_all(
        [
            SqlLatestMetric(
                run_uuid=run_uuid,
                key=key,
                step=step,
                timestamp=timestamp,
                value=value,
                is_nan=is_nan)
            for run_uuid, key, step, timestamp, value, is_nan in all_latest_metrics
        ]
    )
    end = datetime.now()
    print("MIGRATE TIME: {}".format((end - begin).total_seconds()))
    session.commit()


def downgrade():
    pass
