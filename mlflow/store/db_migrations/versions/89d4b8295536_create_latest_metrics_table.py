"""create latest metrics table

Revision ID: 89d4b8295536
Revises: 7ac759974ad8
Create Date: 2019-08-20 11:53:28.178479

"""
import time
import logging

from alembic import op
from sqlalchemy import orm, func, distinct, and_
from sqlalchemy import Column, String, ForeignKey, Float, BigInteger, PrimaryKeyConstraint, Boolean
from mlflow.store.tracking.dbmodels.models import SqlMetric, SqlLatestMetric

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

# revision identifiers, used by Alembic.
revision = "89d4b8295536"
down_revision = "7ac759974ad8"
branch_labels = None
depends_on = None


def _describe_migration_if_necessary(session):
    """
    If the targeted database contains any metric entries, this function emits important,
    database-specific information about the ``create_latest_metrics_table`` migration.
    If the targeted database does *not* contain any metric entries, this output is omitted
    in order to avoid superfluous log output when initializing a new Tracking database.
    """
    num_metric_entries = session.query(SqlMetric).count()
    if num_metric_entries <= 0:
        return

    _logger.warning(
        "**IMPORTANT**: This migration creates a `latest_metrics` table and populates it with the"
        " latest metric entry for each unique (run_id, metric_key) tuple. Latest metric entries are"
        " computed based on step, timestamp, and value. This migration may take a long time for"
        " databases containing a large number of metric entries. Please refer to {readme_link} for"
        " information about this migration, including how to estimate migration size and how to"
        " restore your database to its original state if the migration is unsuccessful. If you"
        " encounter failures while executing this migration, please file a GitHub issue at"
        " {issues_link}.".format(
            readme_link=(
                "https://github.com/mlflow/mlflow/blob/master/mlflow/store/db_migrations/README.md"
                "#89d4b8295536_create_latest_metrics_table"
            ),
            issues_link="https://github.com/mlflow/mlflow/issues",
        )
    )

    num_metric_keys = (
        session.query(SqlMetric.run_uuid, SqlMetric.key)
        .group_by(SqlMetric.run_uuid, SqlMetric.key)
        .count()
    )
    num_runs_containing_metrics = session.query(distinct(SqlMetric.run_uuid)).count()
    _logger.info(
        "This tracking database has {num_metric_entries} total metric entries for {num_metric_keys}"
        " unique metrics across {num_runs} runs.".format(
            num_metric_entries=num_metric_entries,
            num_metric_keys=num_metric_keys,
            num_runs=num_runs_containing_metrics,
        )
    )


def _get_latest_metrics_for_runs(session):
    metrics_with_max_step = (
        session.query(SqlMetric.run_uuid, SqlMetric.key, func.max(SqlMetric.step).label("step"))
        .group_by(SqlMetric.key, SqlMetric.run_uuid)
        .subquery("metrics_with_max_step")
    )
    metrics_with_max_timestamp = (
        session.query(
            SqlMetric.run_uuid,
            SqlMetric.key,
            SqlMetric.step,
            func.max(SqlMetric.timestamp).label("timestamp"),
        )
        .join(
            metrics_with_max_step,
            and_(
                SqlMetric.step == metrics_with_max_step.c.step,
                SqlMetric.run_uuid == metrics_with_max_step.c.run_uuid,
                SqlMetric.key == metrics_with_max_step.c.key,
            ),
        )
        .group_by(SqlMetric.key, SqlMetric.run_uuid, SqlMetric.step)
        .subquery("metrics_with_max_timestamp")
    )
    metrics_with_max_value = (
        session.query(
            SqlMetric.run_uuid,
            SqlMetric.key,
            SqlMetric.step,
            SqlMetric.timestamp,
            func.max(SqlMetric.value).label("value"),
            SqlMetric.is_nan,
        )
        .join(
            metrics_with_max_timestamp,
            and_(
                SqlMetric.timestamp == metrics_with_max_timestamp.c.timestamp,
                SqlMetric.run_uuid == metrics_with_max_timestamp.c.run_uuid,
                SqlMetric.key == metrics_with_max_timestamp.c.key,
                SqlMetric.step == metrics_with_max_timestamp.c.step,
            ),
        )
        .group_by(
            SqlMetric.run_uuid, SqlMetric.key, SqlMetric.step, SqlMetric.timestamp, SqlMetric.is_nan
        )
        .all()
    )
    return metrics_with_max_value


def upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    _describe_migration_if_necessary(session)
    all_latest_metrics = _get_latest_metrics_for_runs(session=session)

    op.create_table(
        SqlLatestMetric.__tablename__,
        Column("key", String(length=250)),
        Column("value", Float(precision=53), nullable=False),
        Column("timestamp", BigInteger, default=lambda: int(time.time())),
        Column("step", BigInteger, default=0, nullable=False),
        Column("is_nan", Boolean, default=False, nullable=False),
        Column("run_uuid", String(length=32), ForeignKey("runs.run_uuid"), nullable=False),
        PrimaryKeyConstraint("key", "run_uuid", name="latest_metric_pk"),
    )

    session.add_all(
        [
            SqlLatestMetric(
                run_uuid=run_uuid,
                key=key,
                step=step,
                timestamp=timestamp,
                value=value,
                is_nan=is_nan,
            )
            for run_uuid, key, step, timestamp, value, is_nan in all_latest_metrics
        ]
    )
    session.commit()

    _logger.info("Migration complete!")


def downgrade():
    op.drop_table(SqlLatestMetric.__tablename__)
