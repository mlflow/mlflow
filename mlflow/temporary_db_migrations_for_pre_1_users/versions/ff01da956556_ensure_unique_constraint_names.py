"""ensure_unique_constraint_names

Revision ID: ff01da956556
Revises: 
Create Date: 2019-05-18 22:58:06.487489

"""
import time

from alembic import op
from sqlalchemy import column, CheckConstraint
from sqlalchemy.orm import relationship, backref
from sqlalchemy import (
    Column, String, Float, ForeignKey, Integer, CheckConstraint,
    BigInteger, PrimaryKeyConstraint)
from sqlalchemy.ext.declarative import declarative_base

# revision identifiers, used by Alembic.
revision = 'ff01da956556'
down_revision = None
branch_labels = None
depends_on = None

# Inline initial runs and experiment table schema for use in migration logic
# Copied from https://github.com/mlflow/mlflow/blob/v0.9.1/mlflow/store/dbmodels/models.py, with
# modifications to substitute constants from MLflow with hard-coded values (e.g. replacing
# SourceType.to_string(SourceType.NOTEBOOK) with the constant "NOTEBOOK").
Base = declarative_base()


SourceTypes = [
    "NOTEBOOK",
    "JOB",
    "LOCAL",
    "UNKNOWN",
    "PROJECT",
]

RunStatusTypes = [
    "SCHEDULED",
    "FAILED",
    "FINISHED",
    "RUNNING",
]


class SqlExperiment(Base):
    """
    DB model for :py:class:`mlflow.entities.Experiment`. These are recorded in ``experiment`` table.
    """
    __tablename__ = 'experiments'

    experiment_id = Column(Integer, autoincrement=True)
    """
    Experiment ID: `Integer`. *Primary Key* for ``experiment`` table.
    """
    name = Column(String(256), unique=True, nullable=False)
    """
    Experiment name: `String` (limit 256 characters). Defined as *Unique* and *Non null* in
                     table schema.
    """
    artifact_location = Column(String(256), nullable=True)
    """
    Default artifact location for this experiment: `String` (limit 256 characters). Defined as
                                                    *Non null* in table schema.
    """
    lifecycle_stage = Column(String(32), default="active")
    """
    Lifecycle Stage of experiment: `String` (limit 32 characters).
                                    Can be either ``active`` (default) or ``deleted``.
    """

    __table_args__ = (
        CheckConstraint(
            lifecycle_stage.in_(["active", "deleted"]),
            name='lifecycle_stage'),
        PrimaryKeyConstraint('experiment_id', name='experiment_pk')
    )

    def __repr__(self):
        return '<SqlExperiment ({}, {})>'.format(self.experiment_id, self.name)


class SqlRun(Base):
    """
    DB model for :py:class:`mlflow.entities.Run`. These are recorded in ``runs`` table.
    """
    __tablename__ = 'runs'

    run_uuid = Column(String(32), nullable=False)
    """
    Run UUID: `String` (limit 32 characters). *Primary Key* for ``runs`` table.
    """
    name = Column(String(250))
    """
    Run name: `String` (limit 250 characters).
    """
    source_type = Column(String(20), default="LOCAL")
    """
    Source Type: `String` (limit 20 characters). Can be one of ``NOTEBOOK``, ``JOB``, ``PROJECT``,
                 ``LOCAL`` (default), or ``UNKNOWN``.
    """
    source_name = Column(String(500))
    """
    Name of source recording the run: `String` (limit 500 characters).
    """
    entry_point_name = Column(String(50))
    """
    Entry-point name that launched the run run: `String` (limit 50 characters).
    """
    user_id = Column(String(256), nullable=True, default=None)
    """
    User ID: `String` (limit 256 characters). Defaults to ``null``.
    """
    status = Column(String(20), default="SCHEDULED")
    """
    Run Status: `String` (limit 20 characters). Can be one of ``RUNNING``, ``SCHEDULED`` (default),
                ``FINISHED``, ``FAILED``.
    """
    start_time = Column(BigInteger, default=int(time.time()))
    """
    Run start time: `BigInteger`. Defaults to current system time.
    """
    end_time = Column(BigInteger, nullable=True, default=None)
    """
    Run end time: `BigInteger`.
    """
    source_version = Column(String(50))
    """
    Source version: `String` (limit 50 characters).
    """
    lifecycle_stage = Column(String(20), default="active")
    """
    Lifecycle Stage of run: `String` (limit 32 characters).
                            Can be either ``active`` (default) or ``deleted``.
    """
    artifact_uri = Column(String(200), default=None)
    """
    Default artifact location for this run: `String` (limit 200 characters).
    """
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'))
    """
    Experiment ID to which this run belongs to: *Foreign Key* into ``experiment`` table.
    """
    experiment = relationship('SqlExperiment', backref=backref('runs', cascade='all'))
    """
    SQLAlchemy relationship (many:one) with :py:class:`mlflow.store.dbmodels.models.SqlExperiment`.
    """

    __table_args__ = (
        CheckConstraint(source_type.in_(SourceTypes), name='source_type'),
        CheckConstraint(status.in_(RunStatusTypes), name='status'),
        CheckConstraint(lifecycle_stage.in_(["active", "deleted"]),
                        name='lifecycle_stage'),
        PrimaryKeyConstraint('run_uuid', name='run_pk')
    )


def upgrade():
    # Use batch mode so that we can run "ALTER TABLE" statements against SQLite
    # databases (see more info at https://alembic.sqlalchemy.org/en/latest/
    # batch.html#running-batch-migrations-for-sqlite-and-other-databases).
    # Also, we directly pass the schema of the table we're modifying to circumvent shortcomings
    # in Alembic's ability to reflect CHECK constraints, as described in
    # https://alembic.sqlalchemy.org/en/latest/batch.html#working-in-offline-mode
    bind = op.get_bind()
    with op.batch_alter_table("experiments", copy_from=SqlExperiment.__table__) as batch_op:
        # We skip running drop_constraint for mysql, because it creates an invalid statement
        # in alembic<=1.0.10
        if bind.engine.name != 'mysql':
            batch_op.drop_constraint(constraint_name='lifecycle_stage', type_="check")
        batch_op.create_check_constraint(
            constraint_name="experiments_lifecycle_stage",
            condition=column('lifecycle_stage').in_(["active", "deleted"])
        )
    with op.batch_alter_table("runs", copy_from=SqlRun.__table__) as batch_op:
        # We skip running drop_constraint for mysql, because it creates an invalid statement
        # in alembic<=1.0.10
        if bind.engine.name != 'mysql':
            batch_op.drop_constraint(constraint_name='lifecycle_stage', type_="check")
        batch_op.create_check_constraint(
            constraint_name="runs_lifecycle_stage",
            condition=column('lifecycle_stage').in_(["active", "deleted"])
        )


def downgrade():
    # Omit downgrade logic for now - we don't currently provide users a command/API for
    # reverting a database migration, instead recommending that they take a database backup
    # before running the migration.
    pass
