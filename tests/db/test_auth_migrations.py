import pytest
import sqlalchemy as sa
from click.testing import CliRunner

from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.server.auth.db import cli
from mlflow.server.auth.db.utils import migrate_if_needed
from mlflow.store.db.utils import _safe_initialize_tables, create_sqlalchemy_engine_with_retry

pytestmark = pytest.mark.notrackingurimock

_AUTH_TABLES = {
    "alembic_version_auth",
    "users",
    "roles",
    "role_permissions",
    "user_role_assignments",
    "experiment_permissions",
    "registered_model_permissions",
    "scorer_permissions",
    "gateway_secret_permissions",
    "gateway_endpoint_permissions",
    "gateway_model_definition_permissions",
    "workspace_permissions",
}


@pytest.fixture
def engine():
    engine = create_sqlalchemy_engine_with_retry(MLFLOW_TRACKING_URI.get())
    # Share the database with the tracking store so that migrations which read tracking
    # tables (e.g. `registered_model_tags` in `f1a2b3c4d5e6`) take their non-trivial path.
    _safe_initialize_tables(engine)
    yield engine
    # Other tests in this directory (e.g. `test_schema.py`) expect only tracking tables.
    metadata = sa.MetaData()
    metadata.reflect(engine, only=lambda name, _: name in _AUTH_TABLES)
    metadata.drop_all(engine)
    engine.dispose()


def _upgrade_with_cli(engine):
    res = CliRunner().invoke(
        cli.upgrade, ["--url", MLFLOW_TRACKING_URI.get()], catch_exceptions=False
    )
    assert res.exit_code == 0, res.output


def _upgrade_on_server_startup(engine):
    migrate_if_needed(engine, "head")


# The timeout turns a MySQL metadata-lock hang in `migrate_if_needed` into a failure.
@pytest.mark.timeout(120, method="thread")
@pytest.mark.parametrize(
    "upgrade", [_upgrade_with_cli, _upgrade_on_server_startup], ids=["cli", "server_startup"]
)
def test_auth_migrations_upgrade_to_head(engine, upgrade):
    upgrade(engine)
    assert _AUTH_TABLES <= set(sa.inspect(engine).get_table_names())
