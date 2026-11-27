"""backfill the kind of experiments that already have traces

Create Date: 2026-11-20

"""

import sqlalchemy as sa
from alembic import op

revision = "3ecac0fa87a0"
down_revision = "f4a9b2c7d1e6"
branch_labels = None
depends_on = None

# The tracking store sets this kind when an experiment without one receives a trace. Tag
# experiments traced before this upgrade, in every workspace, the same way.
_EXPERIMENT_KIND_TAG = "mlflow.experimentKind"
_TRACED_EXPERIMENT_KIND = "genai_development_inferred"


def upgrade():
    experiments = sa.table("experiments", sa.column("experiment_id"))
    experiment_tags = sa.table(
        "experiment_tags", sa.column("key"), sa.column("value"), sa.column("experiment_id")
    )
    existing_tags = experiment_tags.alias("existing_tags")
    trace_info = sa.table("trace_info", sa.column("experiment_id"))

    has_traces = sa.exists().where(trace_info.c.experiment_id == experiments.c.experiment_id)
    has_kind = sa.exists().where(
        existing_tags.c.experiment_id == experiments.c.experiment_id,
        existing_tags.c.key == _EXPERIMENT_KIND_TAG,
    )
    traced_without_kind = sa.select(
        sa.literal(_EXPERIMENT_KIND_TAG, sa.String),
        sa.literal(_TRACED_EXPERIMENT_KIND, sa.String),
        experiments.c.experiment_id,
    ).where(has_traces, ~has_kind)
    op.execute(
        experiment_tags.insert().from_select(["key", "value", "experiment_id"], traced_without_kind)
    )


def downgrade():
    # The backfilled tags are indistinguishable from ones the tracking store set after the
    # upgrade, so they are kept.
    pass
