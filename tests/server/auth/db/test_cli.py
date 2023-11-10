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
            ("alembic_version",),
            ("users",),
            ("experiment_permissions",),
            ("registered_model_permissions",),
        ]
