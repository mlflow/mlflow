from pathlib import Path

from alembic.command import stamp, upgrade
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.engine.base import Engine

INITIAL_REVISION = "8606fa83a998"


def _get_alembic_dir() -> str:
    return Path(__file__).parent / "migrations"


def _get_alembic_config(url: str) -> Config:
    alembic_dir = _get_alembic_dir()
    alembic_ini_path = alembic_dir / "alembic.ini"
    alembic_cfg = Config(alembic_ini_path)
    alembic_cfg.set_main_option("script_location", str(alembic_dir))
    url = url.replace("%", "%%")  # Same as here: https://github.com/mlflow/mlflow/issues/1487
    alembic_cfg.set_main_option("sqlalchemy.url", url)
    return alembic_cfg


def _is_legacy_database(engine: Engine) -> bool:
    """Check if this is a pre-3.6.0 auth database that needs version table migration.

    Returns True if:
    - Auth tables (users, experiment_permissions, etc.) exist
    - alembic_version_auth table does NOT exist

    This indicates a database from MLflow < 3.6.0 that was using the old
    initialization method without Alembic version tracking.
    """
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    has_auth_tables = "users" in existing_tables
    has_version_table = "alembic_version_auth" in existing_tables

    return has_auth_tables and not has_version_table


def _stamp_legacy_database(engine: Engine, revision: str) -> None:
    """Stamp a legacy database with the initial Alembic revision.

    This creates the alembic_version_auth table and marks the database
    as being at the specified revision, avoiding the need to re-run
    the initial migration on databases that already have the tables.
    """
    alembic_cfg = _get_alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.begin() as conn:
        alembic_cfg.attributes["connection"] = conn
        stamp(alembic_cfg, revision)


def migrate(engine: Engine, revision: str) -> None:
    if _is_legacy_database(engine):
        _stamp_legacy_database(engine, INITIAL_REVISION)
        return

    alembic_cfg = _get_alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.begin() as conn:
        alembic_cfg.attributes["connection"] = conn
        upgrade(alembic_cfg, revision)


def migrate_if_needed(engine: Engine, revision: str) -> None:
    if _is_legacy_database(engine):
        _stamp_legacy_database(engine, INITIAL_REVISION)
        return

    alembic_cfg = _get_alembic_config(engine.url.render_as_string(hide_password=False))
    script_dir = ScriptDirectory.from_config(alembic_cfg)
    with engine.begin() as conn:
        context = MigrationContext.configure(conn, opts={"version_table": "alembic_version_auth"})
        if context.get_current_revision() != script_dir.get_current_head():
            upgrade(alembic_cfg, revision)
