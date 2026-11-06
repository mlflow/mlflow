# Cross-dialect (Docker matrix) tests for concurrent skill version registration.
#
# RFC-0008 allocates a skill version number as MAX(version) + 1 and relies on the primary
# key to settle races: concurrent registrations that compute the same number collide, and
# the losing writer recomputes and retries. How a losing writer fails is engine-specific
# (PostgreSQL aborts its transaction, MySQL and SQL Server wait on row locks, SQLite waits
# on its database lock), so this runs on every backend in the database matrix rather than
# only on SQLite.
#
# NOTE: all tests here share one database, so each test works in its own organization
# (the `org` fixture).

import threading
import uuid
from pathlib import Path

import pytest

from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.store.tracking.dbmodels.models import SqlSkillVersion
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture
def store(tmp_path: Path):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    store = SqlAlchemyStore(MLFLOW_TRACKING_URI.get(), artifact_uri.as_uri())
    try:
        yield store
    finally:
        store._dispose_engine()


@pytest.fixture
def org():
    return f"race-{uuid.uuid4().hex[:12]}"


@pytest.mark.parametrize("parent_exists", [True, False], ids=["existing-skill", "new-skill"])
def test_db_backend_concurrent_registrations_get_distinct_versions(
    store, org, monkeypatch, parent_exists
):
    # One writer per allowed attempt, so every writer can succeed within the retry budget
    # even if it loses every race but the last.
    workers = store.CREATE_SKILL_VERSION_RETRIES
    if parent_exists:
        store.create_skill("reviewer", organization=org)

    # Hold every writer's first attempt until all of them have read MAX(version), so all of
    # them target version 1 and workers - 1 of them must lose. For an existing skill they
    # collide on the version row; for a new skill they collide earlier, on the parent row
    # each of them tries to create. Each case exercises a different retry path. Retries
    # skip the barrier.
    original_persist = store._persist_skill_version
    first_attempts = threading.Barrier(workers, timeout=30)
    arrived = set()
    calls = []

    def persist_in_lockstep(*args, **kwargs):
        calls.append(threading.current_thread().name)
        if threading.current_thread().name not in arrived:
            arrived.add(threading.current_thread().name)
            first_attempts.wait()
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(store, "_persist_skill_version", persist_in_lockstep)

    versions = []
    errors = []

    def register():
        try:
            created = store.create_skill_version(
                "reviewer", organization=org, source_type="git", source="https://h/r.git"
            )
        except Exception as e:
            errors.append(e)
        else:
            versions.append(created.version)

    # Daemon threads, so a writer stuck on a database lock fails the assertions below
    # instead of keeping the test process alive; SQL Server waits on locks indefinitely.
    threads = [
        threading.Thread(target=register, name=f"skill-registration-{i}", daemon=True)
        for i in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert sorted(versions) == list(range(1, workers + 1))
    # Only one first attempt could win, so the losers really did collide and retry rather
    # than happening to run one after another.
    assert len(calls) >= 2 * workers - 1
    with store.ManagedSessionMaker() as session:
        stored = (
            session
            .query(SqlSkillVersion.version)
            .filter(SqlSkillVersion.organization == org, SqlSkillVersion.name == "reviewer")
            .all()
        )
    assert sorted(version for (version,) in stored) == list(range(1, workers + 1))
