# Cross-dialect (Docker matrix) tests for the RFC-0008 skill registry schema: they run
# against every backend in the MLflow DB test matrix (SQLite, PostgreSQL, MySQL, MSSQL)
# and prove cascade behavior at the SQL
# layer rather than through application checks. The store layer does not exist yet, so
# these operate directly on the ORM models.
#
# Several tests here intentionally mirror SQLite-only tests in
# tests/store/tracking/test_skill_registry_dbmodels.py: those give fast feedback in the
# normal test suite, while these re-prove the same guarantees on every engine, where
# constraint enforcement (foreign keys, primary keys) and text sort order can differ
# between databases. Each such test names its counterpart.
#
# NOTE: all tests here deliberately share one database, unlike the SQLite twin file, where
# each test can create and discard its own database cheaply. So keep every test to its own
# data: give skills names unique to the test that creates them, and filter
# queries by those names rather than relying on a bare `.all()`. Also remember that
# test_db_backend_migration_downgrade_and_reupgrade drops and recreates these tables in
# that same database mid-run.

from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.orm import Session

from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.tracking.dbmodels.models import (
    SqlSkill,
    SqlSkillAlias,
    SqlSkillTag,
    SqlSkillVersion,
    SqlSkillVersionTag,
)
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


@contextmanager
def session_scope(store, *, commit=True):
    with Session(store.engine) as session:
        if store.engine.dialect.name == "sqlite":
            session.execute(sa.text("PRAGMA foreign_keys = ON"))
        yield session
        if commit:
            session.commit()


def _seed_skill(session, *, organization, name, version=1):
    if session.get(SqlSkill, ("default", organization, name)) is None:
        session.add(SqlSkill(organization=organization, name=name))
    session.add(
        SqlSkillVersion(
            organization=organization, name=name, version=version, source_type="git", source="s.git"
        )
    )


def test_db_backend_cascade_delete_skill(store):
    # Cross-dialect twin of test_cascade_delete_skill_removes_children.
    # Deletes through the ORM (not raw DB constraints), so relationship
    # `cascade="all, delete-orphan"` removes the children.
    # test_db_backend_cascade_is_enforced_by_the_database is the database-level proof.
    with session_scope(store) as session:
        _seed_skill(session, organization="acme", name="cascade-skill")
        session.add(SqlSkillTag(organization="acme", name="cascade-skill", key="k", value="v"))
        session.add(
            SqlSkillVersionTag(
                organization="acme", name="cascade-skill", version=1, key="k", value="v"
            )
        )
        session.add(
            SqlSkillAlias(organization="acme", name="cascade-skill", alias="prod", version=1)
        )
    with session_scope(store) as session:
        session.delete(session.get(SqlSkill, ("default", "acme", "cascade-skill")))
    with session_scope(store, commit=False) as session:
        for model in (SqlSkillVersion, SqlSkillTag, SqlSkillVersionTag, SqlSkillAlias):
            remaining = (
                session.query(model).filter_by(organization="acme", name="cascade-skill").count()
            )
            assert remaining == 0


def test_db_backend_cascade_is_enforced_by_the_database(store):
    # Same as test_db_backend_cascade_delete_skill, but relies on the raw
    # `ON DELETE CASCADE` rule.
    with session_scope(store) as session:
        _seed_skill(session, organization="acme", name="raw-cascade-skill")
        session.add(SqlSkillTag(organization="acme", name="raw-cascade-skill", key="k", value="v"))
        session.add(
            SqlSkillVersionTag(
                organization="acme", name="raw-cascade-skill", version=1, key="k", value="v"
            )
        )
        session.add(
            SqlSkillAlias(organization="acme", name="raw-cascade-skill", alias="prod", version=1)
        )
    # Deleting the parent skill must take every child row with it.
    with session_scope(store) as session:
        session.execute(
            sa.delete(SqlSkill).where(
                SqlSkill.workspace == "default",
                SqlSkill.organization == "acme",
                SqlSkill.name == "raw-cascade-skill",
            )
        )

    with session_scope(store, commit=False) as session:
        for model in (SqlSkillVersion, SqlSkillTag, SqlSkillVersionTag, SqlSkillAlias):
            assert (
                session
                .query(model)
                .filter_by(organization="acme", name="raw-cascade-skill")
                .count()
                == 0
            )


_SKILL_REGISTRY_TABLES = frozenset({
    "skills",
    "skill_versions",
    "skill_tags",
    "skill_version_tags",
    "skill_aliases",
})


def test_db_backend_migration_downgrade_and_reupgrade(store):
    # Cross-dialect twin of test_migration_downgrade_and_reupgrade. FK-aware drop
    # ordering is stricter on MySQL/MSSQL than on SQLite, so this re-proves a clean
    # downgrade on the full matrix. The store fixture leaves the DB at head; downgrade
    # drops this migration's tables, and the finally restores head so the other
    # tests/db tests still see the full schema.
    url = MLFLOW_TRACKING_URI.get()
    config = _get_alembic_config(url)
    assert _SKILL_REGISTRY_TABLES <= set(sa.inspect(store.engine).get_table_names())
    try:
        command.downgrade(config, "b7e2c1a4d9f3")
        assert _SKILL_REGISTRY_TABLES.isdisjoint(set(sa.inspect(store.engine).get_table_names()))
    finally:
        command.upgrade(config, "head")
    assert _SKILL_REGISTRY_TABLES <= set(sa.inspect(store.engine).get_table_names())
