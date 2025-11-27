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


def test_upgrade_from_legacy_database(tmp_path: Path) -> None:
    """Test upgrading from a pre-3.6.0 database that has auth tables but no alembic_version_auth."""
    runner = CliRunner()
    db = tmp_path / "test.db"
    db_url = f"sqlite:///{db}"

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER NOT NULL PRIMARY KEY,
                username VARCHAR(255),
                password_hash VARCHAR(255),
                is_admin BOOLEAN,
                UNIQUE (username)
            )
        """)
        cursor.execute("""
            CREATE TABLE experiment_permissions (
                id INTEGER NOT NULL PRIMARY KEY,
                experiment_id VARCHAR(255) NOT NULL,
                user_id INTEGER NOT NULL,
                permission VARCHAR(255),
                FOREIGN KEY(user_id) REFERENCES users (id),
                UNIQUE (experiment_id, user_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE registered_model_permissions (
                id INTEGER NOT NULL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                user_id INTEGER NOT NULL,
                permission VARCHAR(255),
                FOREIGN KEY(user_id) REFERENCES users (id),
                UNIQUE (name, user_id)
            )
        """)
        cursor.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
            ("testuser", "hash123", True),
        )
        conn.commit()

    res = runner.invoke(cli.upgrade, ["--url", db_url], catch_exceptions=False)
    assert res.exit_code == 0, res.output

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {t[0] for t in cursor.fetchall()}

        cursor.execute("SELECT version_num FROM alembic_version_auth;")
        version = cursor.fetchone()

        cursor.execute("SELECT username, is_admin FROM users;")
        user = cursor.fetchone()

    assert "alembic_version_auth" in tables
    assert "users" in tables
    assert "experiment_permissions" in tables
    assert "registered_model_permissions" in tables
    assert version[0] == "8606fa83a998"
    assert user == ("testuser", 1)
