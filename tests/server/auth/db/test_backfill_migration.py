from pathlib import Path

import pytest
from alembic.command import downgrade as alembic_downgrade
from sqlalchemy import create_engine, text

from mlflow.server.auth.db.utils import _get_alembic_config, migrate
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

M1_REVISION = "c3d4e5f6a7b8"
M2_REVISION = "d4e5f6a7b8c9"


def _downgrade(engine, revision: str) -> None:
    cfg = _get_alembic_config(engine.url.render_as_string(hide_password=False))
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        alembic_downgrade(cfg, revision)


@pytest.fixture
def engine(tmp_path: Path):
    db = tmp_path / "backfill.db"
    eng = create_engine(f"sqlite:///{db}")
    yield eng
    eng.dispose()


def _seed_legacy_state(
    engine, users: list[tuple[int, str]], grants: list[dict[str, object]]
) -> None:
    with engine.begin() as conn:
        for user_id, username in users:
            conn.execute(
                text(
                    "INSERT INTO users (id, username, password_hash, is_admin) "
                    "VALUES (:id, :u, 'hash', 0)"
                ),
                {"id": user_id, "u": username},
            )
        for g in grants:
            table = g.pop("table")
            cols = ", ".join(g.keys())
            placeholders = ", ".join(f":{k}" for k in g.keys())
            conn.execute(text(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"), g)


def _fetch_role_permissions(engine) -> list[dict[str, object]]:
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT r.workspace, r.name, rp.resource_type, rp.resource_pattern, rp.permission "
                "FROM role_permissions rp "
                "JOIN roles r ON r.id = rp.role_id "
                "ORDER BY r.workspace, r.name, rp.resource_type, rp.resource_pattern"
            )
        ).mappings()
        return [dict(row) for row in rows]


def _fetch_assignments(engine) -> set[tuple[int, str]]:
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                "SELECT a.user_id, r.name "
                "FROM user_role_assignments a JOIN roles r ON r.id = a.role_id"
            )
        )
        return {(r[0], r[1]) for r in rows}


def test_backfill_mirrors_every_legacy_grant(engine):
    migrate(engine, M1_REVISION)
    _seed_legacy_state(
        engine,
        users=[(1, "alice"), (2, "bob")],
        grants=[
            {
                "table": "experiment_permissions",
                "user_id": 1,
                "experiment_id": "exp1",
                "permission": "READ",
            },
            {
                "table": "experiment_permissions",
                "user_id": 2,
                "experiment_id": "exp1",
                "permission": "EDIT",
            },
            {
                "table": "registered_model_permissions",
                "user_id": 1,
                "workspace": "ws-a",
                "name": "model_x",
                "permission": "MANAGE",
            },
            {
                "table": "scorer_permissions",
                "user_id": 1,
                "experiment_id": "exp1",
                "scorer_name": "judge_a",
                "permission": "USE",
            },
            {
                "table": "gateway_secret_permissions",
                "user_id": 2,
                "secret_id": "sec1",
                "permission": "READ",
            },
            {
                "table": "gateway_endpoint_permissions",
                "user_id": 1,
                "endpoint_id": "ep1",
                "permission": "USE",
            },
            {
                "table": "gateway_model_definition_permissions",
                "user_id": 1,
                "model_definition_id": "md1",
                "permission": "READ",
            },
            {
                "table": "workspace_permissions",
                "user_id": 2,
                "workspace": "ws-a",
                "permission": "MANAGE",
            },
        ],
    )

    migrate(engine, M2_REVISION)

    rps = _fetch_role_permissions(engine)
    # Convert to a lookup set of (workspace, role_name, type, pattern, perm).
    triples = {
        (r["workspace"], r["name"], r["resource_type"], r["resource_pattern"], r["permission"])
        for r in rps
    }
    default = DEFAULT_WORKSPACE_NAME
    assert (default, "__user_1__", "experiment", "exp1", "READ") in triples
    assert (default, "__user_2__", "experiment", "exp1", "EDIT") in triples
    assert ("ws-a", "__user_1__", "registered_model", "model_x", "MANAGE") in triples
    assert (default, "__user_1__", "scorer", "exp1/judge_a", "USE") in triples
    assert (default, "__user_2__", "gateway_secret", "sec1", "READ") in triples
    assert (default, "__user_1__", "gateway_endpoint", "ep1", "USE") in triples
    assert (default, "__user_1__", "gateway_model_definition", "md1", "READ") in triples
    assert ("ws-a", "__user_2__", "*", "*", "MANAGE") in triples

    # Each user is assigned to every synthetic role created for them.
    assignments = _fetch_assignments(engine)
    assert (1, "__user_1__") in assignments
    assert (2, "__user_2__") in assignments


def test_backfill_is_idempotent_under_downgrade_upgrade(engine):
    # Legacy grants stay in place across the downgrade/upgrade cycle, so re-running the
    # backfill must produce the same role_permissions state without duplicating rows.
    migrate(engine, M1_REVISION)
    _seed_legacy_state(
        engine,
        users=[(1, "alice")],
        grants=[
            {
                "table": "experiment_permissions",
                "user_id": 1,
                "experiment_id": "exp1",
                "permission": "READ",
            },
            {
                "table": "workspace_permissions",
                "user_id": 1,
                "workspace": "ws-a",
                "permission": "EDIT",
            },
        ],
    )
    migrate(engine, M2_REVISION)
    snapshot_a = _fetch_role_permissions(engine)
    assert snapshot_a  # sanity

    _downgrade(engine, M1_REVISION)
    migrate(engine, M2_REVISION)
    snapshot_b = _fetch_role_permissions(engine)

    assert snapshot_a == snapshot_b


def test_backfill_downgrade_removes_synthetic_roles(engine):
    migrate(engine, M1_REVISION)
    # Also create a user-defined role to make sure downgrade doesn't nuke it.
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO roles (name, workspace, description) "
                "VALUES ('viewer', 'ws-a', 'read only')"
            )
        )
    _seed_legacy_state(
        engine,
        users=[(1, "alice")],
        grants=[
            {
                "table": "experiment_permissions",
                "user_id": 1,
                "experiment_id": "exp1",
                "permission": "READ",
            },
        ],
    )
    migrate(engine, M2_REVISION)

    # Downgrade back to M1 — synthetic per-user roles should be removed, user-defined
    # roles should survive.
    _downgrade(engine, M1_REVISION)

    with engine.begin() as conn:
        names = [r[0] for r in conn.execute(text("SELECT name FROM roles"))]
    assert "viewer" in names
    assert all(not (n.startswith("__user_") and n.endswith("__")) for n in names)
