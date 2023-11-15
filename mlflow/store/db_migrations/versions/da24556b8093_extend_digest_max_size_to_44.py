"""extend digest max size to 44

Revision ID: da24556b8093
Revises: acf3f17fdcc7
Create Date: 2023-11-15 20:54:25.104933

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "da24556b8093"
down_revision = "acf3f17fdcc7"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('new_datasets',
                    sa.Column('dataset_uuid', sa.String(36), nullable=False),
                    sa.Column('experiment_id', sa.Integer, nullable=False),
                    sa.Column('name', sa.String(500), nullable=False),
                    sa.Column('digest', sa.String(44), nullable=False),
                    sa.Column('dataset_source_type', sa.String(36), nullable=False),
                    sa.Column('dataset_source', sa.Text, nullable=False),
                    sa.Column('dataset_schema', sa.Text),
                    sa.Column('dataset_profile', sa.Text),
                    sa.PrimaryKeyConstraint('experiment_id', 'name', 'digest'),
                    sa.ForeignKeyConstraint(['experiment_id'], ['experiments.experiment_id'])
                    )
    op.execute('INSERT INTO new_datasets SELECT * FROM datasets')
    op.drop_table('datasets')
    op.rename_table('new_datasets', 'datasets')

def downgrade():
    pass
