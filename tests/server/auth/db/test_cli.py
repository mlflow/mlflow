import sqlite3
from pathlib import Path
from urllib.parse import quote

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

    # The legacy per-resource permission tables are created by earlier migrations
    # and intentionally retained by ``e5f6a7b8c9d0`` so operators can still roll
    # back without restoring from backup. A future migration will drop them.
    assert sorted(tables) == sorted([
        ("alembic_version_auth",),
        ("users",),
        ("roles",),
        ("role_permissions",),
        ("user_role_assignments",),
        ("experiment_permissions",),
        ("registered_model_permissions",),
        ("scorer_permissions",),
        ("gateway_secret_permissions",),
        ("gateway_endpoint_permissions",),
        ("gateway_model_definition_permissions",),
        ("workspace_permissions",),
    ])


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
    assert "roles" in tables
    assert "role_permissions" in tables
    assert "user_role_assignments" in tables
    assert "experiments" in tables
    assert "runs" in tables
    # Legacy per-resource permission tables are intentionally retained by the
    # ``e5f6a7b8c9d0`` migration so operators can still roll back without
    # restoring from backup.
    assert "experiment_permissions" in tables
    assert "registered_model_permissions" in tables
    assert "scorer_permissions" in tables
    assert "gateway_secret_permissions" in tables
    assert "gateway_endpoint_permissions" in tables
    assert "gateway_model_definition_permissions" in tables
    assert "workspace_permissions" in tables


def test_upgrade_from_legacy_database(tmp_path: Path) -> None:
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
    assert "roles" in tables
    assert "role_permissions" in tables
    assert "user_role_assignments" in tables
    # Legacy tables are intentionally retained by ``e5f6a7b8c9d0`` so operators
    # can still roll back without restoring from backup. A future migration
    # will drop them once the simplified RBAC model has bedded in.
    assert "experiment_permissions" in tables
    assert "scorer_permissions" in tables
    assert "registered_model_permissions" in tables
    assert "workspace_permissions" in tables
    assert version[0] == "e5f6a7b8c9d0"
    assert user == ("testuser", 1)


def test_upgrade_filters_legacy_rows_per_simplified_model(tmp_path: Path) -> None:
    """Migration drops resource-scope ``NO_PERMISSIONS`` rows and rewrites
    workspace-scope ``READ`` / ``EDIT`` rows to ``USE``. Other values pass through.

    Walks the migration chain in two steps: first up to the revision **before**
    the backfill so the legacy tables exist with the expected schema, seeds
    legacy data, then runs the backfill migration. This avoids pre-creating
    legacy tables by hand (which would collide with earlier migrations that
    create them).
    """
    runner = CliRunner()
    db = tmp_path / "test.db"
    db_url = f"sqlite:///{db}"

    # Walk to the revision that immediately precedes the backfill migration.
    res = runner.invoke(
        cli.upgrade,
        ["--url", db_url, "--revision", "c3d4e5f6a7b8"],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO users (id, username, password_hash, is_admin) VALUES (?, ?, ?, ?)",
            [
                (1, "alice", "h", False),
                (2, "bob", "h", False),
                (3, "carol", "h", False),
                (4, "dan", "h", False),
            ],
        )
        # Resource-scope rows: NO_PERMISSIONS dropped, READ / MANAGE preserved.
        cursor.executemany(
            "INSERT INTO experiment_permissions (experiment_id, user_id, permission)"
            " VALUES (?, ?, ?)",
            [
                ("exp-1", 1, "READ"),
                ("exp-2", 2, "NO_PERMISSIONS"),
                ("exp-3", 3, "MANAGE"),
            ],
        )
        # Workspace-scope rows: NO_PERMISSIONS dropped, READ / EDIT rewritten
        # to USE, USE / MANAGE preserved.
        cursor.executemany(
            "INSERT INTO workspace_permissions (workspace, user_id, permission) VALUES (?, ?, ?)",
            [
                ("ws-default", 1, "READ"),
                ("ws-default", 2, "EDIT"),
                ("ws-default", 3, "NO_PERMISSIONS"),
                ("ws-default", 4, "MANAGE"),
            ],
        )
        conn.commit()

    # Run the backfill migration.
    res = runner.invoke(cli.upgrade, ["--url", db_url], catch_exceptions=False)
    assert res.exit_code == 0, res.output

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT u.username, rp.resource_type, rp.resource_pattern, rp.permission "
            "FROM role_permissions rp "
            "JOIN roles r ON r.id = rp.role_id "
            "JOIN user_role_assignments ura ON ura.role_id = r.id "
            "JOIN users u ON u.id = ura.user_id "
            "ORDER BY u.username, rp.resource_type, rp.resource_pattern"
        )
        rows = cursor.fetchall()

    # alice: experiment READ preserved, workspace READ rewritten to USE
    # bob:   experiment NO_PERMISSIONS skipped, workspace EDIT rewritten to USE
    # carol: experiment MANAGE preserved, workspace NO_PERMISSIONS skipped
    # dan:   workspace MANAGE preserved
    assert rows == [
        ("alice", "*", "*", "USE"),
        ("alice", "experiment", "exp-1", "READ"),
        ("bob", "*", "*", "USE"),
        ("carol", "experiment", "exp-3", "MANAGE"),
        ("dan", "*", "*", "MANAGE"),
    ]


def test_upgrade_encodes_scorer_names_with_special_characters(tmp_path: Path) -> None:
    """Scorer names may contain ``/`` (validated only against empty/whitespace)
    and other URL-special chars. The migration URL-encodes the name component
    of the scorer compound key so the resulting ``resource_pattern`` is
    unambiguous, and the runtime store uses the same encoding — verify the
    round-trip survives the migration.
    """
    runner = CliRunner()
    db = tmp_path / "test.db"
    db_url = f"sqlite:///{db}"

    res = runner.invoke(
        cli.upgrade,
        ["--url", db_url, "--revision", "c3d4e5f6a7b8"],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output

    scorer_names = [
        "plain-name",
        "with/slash",
        "with%percent",
        "with space and #hash",
    ]
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (id, username, password_hash, is_admin) VALUES (?, ?, ?, ?)",
            (1, "alice", "h", False),
        )
        cursor.executemany(
            "INSERT INTO scorer_permissions (experiment_id, scorer_name, user_id, permission)"
            " VALUES (?, ?, ?, ?)",
            [("exp-42", name, 1, "READ") for name in scorer_names],
        )
        conn.commit()

    res = runner.invoke(cli.upgrade, ["--url", db_url], catch_exceptions=False)
    assert res.exit_code == 0, res.output

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT rp.resource_pattern FROM role_permissions rp "
            "WHERE rp.resource_type = 'scorer' "
            "ORDER BY rp.resource_pattern"
        )
        patterns = [row[0] for row in cursor.fetchall()]

    expected = sorted(f"exp-42/{quote(name, safe='')}" for name in scorer_names)
    assert patterns == expected
