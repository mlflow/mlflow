"""alter metrics primary key

Revision ID: 53d92e2caa1b
Revises: f6c994d15571
Create Date: 2019-04-16 22:11:04.609222

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '53d92e2caa1b'
down_revision = 'f6c994d15571'
branch_labels = None
depends_on = None


def upgrade():
    # Stolen from https://stackoverflow.com/a/48091035
    # Drop primary key constraint. Note the CASCASE clause - this deletes the foreign key constraint.
    op.execute('ALTER TABLE user DROP CONSTRAINT pk_user CASCADE')
    # Change primary key type
    op.alter_column('user', 'id', existing_type=sa.Integer, type_=sa.VARCHAR(length=25))
    op.alter_column('roles_users', 'user_id', existing_type=sa.Integer, type_=sa.VARCHAR(length=25))
    # Re-create the primary key constraint
    op.create_primary_key('pk_user', 'user', ['id'])
    # Re-create the foreign key constraint
    op.create_foreign_key('fk_roles_user_user_id_user', 'roles_users', 'user', ['user_id'], ['id'], ondelete='CASCADE')


def downgrade():
    pass
