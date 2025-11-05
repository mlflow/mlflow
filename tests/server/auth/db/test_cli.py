import sqlite3
from pathlib import Path

from click.testing import CliRunner

from mlflow.server.auth.db import cli


def test_upgrade(tmp_path: Path) -> None:
    runner = CliRunner()
    db = tmp_path / "test.db"
    res = runner.invoke(cli.upgrade, ["--url", f"sqlite:///{db}"], catch_exceptions=False)
    assert res.exit_code == 0, res.output

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

    assert tables == [
        ("alembic_version_auth",),
        ("users",),
        ("experiment_permissions",),
        ("registered_model_permissions",),
        ("scorer_permissions",),
    ]


def test_auth_and_tracking_store_coexist(tmp_path: Path) -> None:
    from mlflow.store.db.utils import _safe_initialize_tables, create_sqlalchemy_engine_with_retry

    runner = CliRunner()
    db = tmp_path / "test.db"
    db_url = f"sqlite:///{db}"

    tracking_engine = create_sqlalchemy_engine_with_retry(db_url)
    _safe_initialize_tables(tracking_engine)

    res = runner.invoke(cli.upgrade, ["--url", db_url], catch_exceptions=False)
    assert res.exit_code == 0, res.output

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {t[0] for t in cursor.fetchall()}

    assert "alembic_version" in tables
    assert "alembic_version_auth" in tables
    assert "users" in tables
    assert "experiment_permissions" in tables
    assert "registered_model_permissions" in tables
    assert "scorer_permissions" in tables
    assert "experiments" in tables
    assert "runs" in tables
