"""migrate user column to tags

Revision ID: 90e64c465722
Revises: 451aebb31d03
Create Date: 2019-05-29 10:43:52.919427

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import orm, Column, Integer, String, ForeignKey, PrimaryKeyConstraint
from sqlalchemy.orm import relationship, backref, declarative_base
from mlflow.utils.mlflow_tags import MLFLOW_USER

# revision identifiers, used by Alembic.
revision = "90e64c465722"
down_revision = "451aebb31d03"
branch_labels = None
depends_on = None


Base = declarative_base()


class SqlRun(Base):
    __tablename__ = "runs"
    run_uuid = Column(String(32), nullable=False)
    user_id = Column(String(256), nullable=True, default=None)
    experiment_id = Column(Integer)

    __table_args__ = (PrimaryKeyConstraint("experiment_id", name="experiment_pk"),)


class SqlTag(Base):
    __tablename__ = "tags"
    key = Column(String(250))
    value = Column(String(250), nullable=True)
    run_uuid = Column(String(32), ForeignKey("runs.run_uuid"))
    run = relationship("SqlRun", backref=backref("tags", cascade="all"))

    __table_args__ = (PrimaryKeyConstraint("key", "run_uuid", name="tag_pk"),)


def upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)
    runs = session.query(SqlRun).all()
    for run in runs:
        if not run.user_id:
            continue

        tag_exists = False
        for tag in run.tags:
            if tag.key == MLFLOW_USER:
                tag_exists = True
        if tag_exists:
            continue

        session.merge(SqlTag(run_uuid=run.run_uuid, key=MLFLOW_USER, value=run.user_id))
    session.commit()


def downgrade():
    pass
