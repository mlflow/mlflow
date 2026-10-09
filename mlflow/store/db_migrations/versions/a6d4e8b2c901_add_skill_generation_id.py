"""Add a stable identifier for each Skill parent instance.

Create Date: 2026-10-09
"""

import uuid

import sqlalchemy as sa
from alembic import op

revision = "a6d4e8b2c901"
down_revision = "e7d1f4b2a9c6"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("skills", sa.Column("generation_id", sa.String(32), nullable=True))
    skills = sa.table(
        "skills",
        sa.column("workspace", sa.String(63)),
        sa.column("organization", sa.String(64)),
        sa.column("name", sa.String(128)),
        sa.column("generation_id", sa.String(32)),
    )
    connection = op.get_bind()
    parents = connection.execute(
        sa.select(skills.c.workspace, skills.c.organization, skills.c.name)
    ).all()
    for workspace, organization, name in parents:
        connection.execute(
            skills
            .update()
            .where(
                skills.c.workspace == workspace,
                skills.c.organization == organization,
                skills.c.name == name,
            )
            .values(generation_id=uuid.uuid4().hex)
        )
    with op.batch_alter_table("skills") as batch_op:
        batch_op.alter_column("generation_id", existing_type=sa.String(32), nullable=False)


def downgrade():
    with op.batch_alter_table("skills") as batch_op:
        batch_op.drop_column("generation_id")
