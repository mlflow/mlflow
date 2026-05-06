import sqlite3
from pathlib import Path
from urllib.parse import quote

from alembic.command import downgrade as alembic_downgrade
from click.testing import CliRunner

from mlflow.server.auth.db import cli
from mlflow.server.auth.db.utils import _get_alembic_config


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
    """Migration drops resource-scope ``NO_PERMISSIONS`` rows; workspace-scope
    ``READ`` rewrites to a single ``USE`` grant; workspace-scope ``EDIT`` fans
    out to ``USE`` plus a type-wildcard ``EDIT`` grant on every concrete
    resource type; other values pass through.

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
        # Workspace-scope rows: NO_PERMISSIONS dropped; READ rewrites to a
        # single USE row; EDIT fans out to USE + per-resource-type EDIT;
        # USE / MANAGE migrate unchanged.
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
    # bob:   experiment NO_PERMISSIONS skipped, workspace EDIT fans out to
    #        ('workspace','*','USE') + ('<each resource_type>','*','EDIT')
    # carol: experiment MANAGE preserved, workspace NO_PERMISSIONS skipped
    # dan:   workspace MANAGE preserved
    assert rows == [
        ("alice", "experiment", "exp-1", "READ"),
        ("alice", "workspace", "*", "USE"),
        ("bob", "experiment", "*", "EDIT"),
        ("bob", "gateway_endpoint", "*", "EDIT"),
        ("bob", "gateway_model_definition", "*", "EDIT"),
        ("bob", "gateway_secret", "*", "EDIT"),
        ("bob", "registered_model", "*", "EDIT"),
        ("bob", "scorer", "*", "EDIT"),
        ("bob", "workspace", "*", "USE"),
        ("carol", "experiment", "exp-3", "MANAGE"),
        ("dan", "workspace", "*", "MANAGE"),
    ]


def test_upgrade_workspace_edit_fans_out_to_per_type_grants(tmp_path: Path) -> None:
    """Workspace ``EDIT`` had no single equivalent in the simplified two-tier
    model (workspace tier is USE / MANAGE only). To preserve the user's
    pre-simplification per-resource EDIT capability, the migration emits one
    workspace-wide ``USE`` grant (for visibility + create) plus a type-wildcard
    ``EDIT`` grant on every concrete resource type. Together those reproduce
    the legacy "EDIT on every resource in this workspace" behavior without
    smuggling EDIT into the workspace tier itself.

    This test is a focused pin for the fan-out so a future migration tweak
    that drops a resource type (or collapses EDIT back to USE) is caught.
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

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (id, username, password_hash, is_admin) VALUES (?, ?, ?, ?)",
            (1, "alice", "h", False),
        )
        cursor.execute(
            "INSERT INTO workspace_permissions (workspace, user_id, permission) VALUES (?, ?, ?)",
            ("ws-team-a", 1, "EDIT"),
        )
        conn.commit()

    res = runner.invoke(cli.upgrade, ["--url", db_url], catch_exceptions=False)
    assert res.exit_code == 0, res.output

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT r.workspace, rp.resource_type, rp.resource_pattern, rp.permission "
            "FROM role_permissions rp "
            "JOIN roles r ON r.id = rp.role_id "
            "JOIN user_role_assignments ura ON ura.role_id = r.id "
            "JOIN users u ON u.id = ura.user_id "
            "WHERE u.username = 'alice' "
            "ORDER BY rp.resource_type, rp.resource_pattern"
        )
        rows = cursor.fetchall()

    # All seven grants land in alice's synthetic role for ``ws-team-a``.
    # Sorted by ``rp.resource_type`` then ``rp.resource_pattern`` (the query's
    # ORDER BY clause) — so per-type EDIT rows come first alphabetically and
    # the workspace-wide USE anchor comes last.
    assert rows == [
        ("ws-team-a", "experiment", "*", "EDIT"),
        ("ws-team-a", "gateway_endpoint", "*", "EDIT"),
        ("ws-team-a", "gateway_model_definition", "*", "EDIT"),
        ("ws-team-a", "gateway_secret", "*", "EDIT"),
        ("ws-team-a", "registered_model", "*", "EDIT"),
        ("ws-team-a", "scorer", "*", "EDIT"),
        ("ws-team-a", "workspace", "*", "USE"),
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


def _alembic_downgrade(db_url: str, revision: str) -> None:
    import sqlalchemy

    engine = sqlalchemy.create_engine(db_url)
    cfg = _get_alembic_config(db_url)
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        alembic_downgrade(cfg, revision)
    engine.dispose()


def test_upgrade_then_downgrade_then_upgrade_is_idempotent(tmp_path: Path) -> None:
    """End-to-end exercise of the ``e5f6a7b8c9d0`` migration's upgrade/downgrade
    pair. Seeds legacy rows of every kind, runs upgrade, downgrades back to the
    pre-backfill revision, and re-runs upgrade. Pins three invariants:

    1. ``downgrade()`` removes every synthetic ``__user_<id>__`` role plus its
       ``role_permissions`` and ``user_role_assignments`` rows, but leaves the
       legacy per-resource tables untouched (rollback safety — operators can
       resume on the legacy code path without restoring from backup).
    2. The legacy table contents survive the round trip byte-for-byte, so
       re-running ``upgrade()`` produces the same ``role_permissions`` output as
       the first run (idempotent backfill).
    3. The downgrade does *not* delete admin-managed roles whose names happen
       to start/end with underscores (regression for the ``__user_admin__``
       false-positive risk that ``_SYNTHETIC_ROLE_NAME_RE`` guards against).
    """
    runner = CliRunner()
    db = tmp_path / "test.db"
    db_url = f"sqlite:///{db}"

    # Step to the revision immediately before the backfill, then seed legacy data.
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
            ],
        )
        cursor.executemany(
            "INSERT INTO experiment_permissions (experiment_id, user_id, permission)"
            " VALUES (?, ?, ?)",
            [("exp-1", 1, "READ"), ("exp-2", 2, "MANAGE")],
        )
        cursor.executemany(
            "INSERT INTO registered_model_permissions (workspace, name, user_id, permission)"
            " VALUES (?, ?, ?, ?)",
            [("ws-default", "model-1", 1, "EDIT")],
        )
        cursor.executemany(
            "INSERT INTO scorer_permissions (experiment_id, scorer_name, user_id, permission)"
            " VALUES (?, ?, ?, ?)",
            [("exp-1", "scorer-a", 1, "USE")],
        )
        cursor.executemany(
            "INSERT INTO workspace_permissions (workspace, user_id, permission) VALUES (?, ?, ?)",
            [("ws-default", 1, "READ"), ("ws-default", 2, "MANAGE")],
        )
        # An admin-managed role whose name starts/ends with underscores but
        # is not a synthetic per-user role. Downgrade must leave it alone.
        cursor.execute(
            "INSERT INTO roles (id, workspace, name) VALUES (?, ?, ?)",
            (999, "ws-default", "__user_admin__"),
        )
        conn.commit()

    # Snapshot legacy state pre-upgrade so we can verify it survives.
    legacy_snapshot = _snapshot_legacy_tables(db)

    # First upgrade.
    res = runner.invoke(cli.upgrade, ["--url", db_url], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    rp_after_first_upgrade = _snapshot_role_permissions(db)
    assert rp_after_first_upgrade, "first upgrade produced no role_permissions rows"

    # Sanity: legacy tables still hold their pre-upgrade contents.
    assert _snapshot_legacy_tables(db) == legacy_snapshot

    # Downgrade one step (back to the pre-backfill revision).
    _alembic_downgrade(db_url, "c3d4e5f6a7b8")

    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM roles WHERE name LIKE '__user_%'")
        surviving_underscore_roles = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) FROM role_permissions")
        rp_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM user_role_assignments")
        ura_count = cursor.fetchone()[0]

    # All synthetic roles are gone; the admin role with the lookalike name stays.
    assert surviving_underscore_roles == [(999, "__user_admin__")]
    # Synthetic roles owned every role_permissions / user_role_assignments row, so
    # both should now be empty.
    assert rp_count == 0
    assert ura_count == 0
    # Legacy contents survive the downgrade — that's the rollback contract.
    assert _snapshot_legacy_tables(db) == legacy_snapshot

    # Re-run upgrade. Output must match the first run row-for-row.
    res = runner.invoke(cli.upgrade, ["--url", db_url], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    assert _snapshot_role_permissions(db) == rp_after_first_upgrade


def _snapshot_role_permissions(db: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT u.username, rp.resource_type, rp.resource_pattern, rp.permission "
            "FROM role_permissions rp "
            "JOIN roles r ON r.id = rp.role_id "
            "JOIN user_role_assignments ura ON ura.role_id = r.id "
            "JOIN users u ON u.id = ura.user_id "
            "ORDER BY u.username, rp.resource_type, rp.resource_pattern, rp.permission"
        )
        return cursor.fetchall()


def _snapshot_legacy_tables(db: Path) -> dict[str, list[tuple[object, ...]]]:
    queries = {
        "experiment_permissions": (
            "SELECT experiment_id, user_id, permission FROM experiment_permissions "
            "ORDER BY experiment_id, user_id"
        ),
        "registered_model_permissions": (
            "SELECT workspace, name, user_id, permission FROM registered_model_permissions "
            "ORDER BY workspace, name, user_id"
        ),
        "scorer_permissions": (
            "SELECT experiment_id, scorer_name, user_id, permission FROM scorer_permissions "
            "ORDER BY experiment_id, scorer_name, user_id"
        ),
        "workspace_permissions": (
            "SELECT workspace, user_id, permission FROM workspace_permissions "
            "ORDER BY workspace, user_id"
        ),
    }
    snapshot: dict[str, list[tuple[object, ...]]] = {}
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        for table, query in queries.items():
            cursor.execute(query)
            snapshot[table] = cursor.fetchall()
    return snapshot
