"""migrate user column to tags

Revision ID: 90e64c465722
Revises: 451aebb31d03
Create Date: 2019-05-29 10:43:52.919427

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import orm
from mlflow.store.dbmodels.models import SqlRun, SqlTag
from mlflow.utils.mlflow_tags import MLFLOW_USER

# revision identifiers, used by Alembic.
revision = '90e64c465722'
down_revision = '451aebb31d03'
branch_labels = None
depends_on = None


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
