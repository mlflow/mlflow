# Cross-dialect (Docker matrix) tests for the RFC-0008 skill registry schema: they run
# against every backend in the MLflow DB test matrix (SQLite, PostgreSQL, MySQL, MSSQL).
# They work on the ORM models and the migration rather than the store: key widths, the
# ORM's cascade on delete, and downgrading and re-upgrading the migration. The database's
# own cascade, without the ORM, is checked in
# tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_skill_registry_constraints.py,
# and the store's behavior on every engine in tests/store/tracking/sqlalchemy_store. The
# database CI job runs both.
#
# NOTE: all tests here deliberately share one database, unlike the store tests, where
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
from alembic.script import ScriptDirectory
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
    # Deletes through the ORM (not raw DB constraints), so relationship
    # `cascade="all, delete-orphan"` removes the children. The database-level proof is
    # test_database_cascades_skill_deletion_to_all_children in
    # tests/store/tracking/sqlalchemy_store/test_sqlalchemy_store_skill_registry_constraints.py.
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


def _widest(model, column, fill):
    return fill * getattr(model, column).type.length


def test_db_backend_keys_accept_values_at_their_declared_maximum(store):
    # https://github.com/mlflow/rfcs/pull/47 sized every composite key to fit each engine's
    # index limit. MySQL enforces that when the migration creates the index, but SQL Server
    # only warns then, and rejects a row once its key actually exceeds 900 bytes. So insert
    # every table's key columns at their declared maximum and read them back unchanged,
    # which also catches a column silently truncating a value. Widths come from the models,
    # so widening a column later is covered too.
    workspace = _widest(SqlSkill, "workspace", "w")
    organization = _widest(SqlSkill, "organization", "o")
    skill = _widest(SqlSkill, "name", "s")
    tag_key = _widest(SqlSkillTag, "key", "k")
    alias = _widest(SqlSkillAlias, "alias", "a")
    digest = _widest(SqlSkillVersion, "digest", "f")

    skill_key = {"workspace": workspace, "organization": organization, "name": skill}
    with session_scope(store) as session:
        session.add(SqlSkill(**skill_key))
        session.add(
            SqlSkillVersion(
                **skill_key, version=1, source_type="git", source="s.git", digest=digest
            )
        )
        session.add(SqlSkillTag(**skill_key, key=tag_key, value="v"))
        session.add(SqlSkillVersionTag(**skill_key, version=1, key=tag_key, value="v"))
        session.add(SqlSkillAlias(**skill_key, alias=alias, version=1))

    with session_scope(store, commit=False) as session:
        assert session.get(SqlSkill, (workspace, organization, skill)) is not None
        stored = session.get(SqlSkillVersion, (workspace, organization, skill, 1))
        assert stored.digest == digest
        assert session.get(SqlSkillTag, (workspace, organization, skill, tag_key)) is not None
        assert session.get(SqlSkillVersionTag, (workspace, organization, skill, 1, tag_key))
        assert session.get(SqlSkillAlias, (workspace, organization, skill, alias)) is not None


_SKILL_REGISTRY_TABLES = frozenset({
    "skills",
    "skill_versions",
    "skill_tags",
    "skill_version_tags",
    "skill_aliases",
})


def _skill_registry_migration_parent(config):
    # Read from the migration rather than hardcoded, so the downgrade undoes only this
    # migration even after other migrations are added beneath it.
    return ScriptDirectory.from_config(config).get_revision("e7d1f4b2a9c6").down_revision


def test_db_backend_migration_downgrade_and_reupgrade(store):
    # FK-aware drop ordering is stricter on MySQL/MSSQL than on SQLite, so this proves a
    # clean downgrade on the full matrix. The store fixture leaves the DB at head; downgrade
    # drops this migration's tables, and the finally restores head so the other
    # tests/db tests still see the full schema.
    url = MLFLOW_TRACKING_URI.get()
    config = _get_alembic_config(url)
    assert _SKILL_REGISTRY_TABLES <= set(sa.inspect(store.engine).get_table_names())
    try:
        command.downgrade(config, _skill_registry_migration_parent(config))
        assert _SKILL_REGISTRY_TABLES.isdisjoint(set(sa.inspect(store.engine).get_table_names()))
    finally:
        command.upgrade(config, "head")
    assert _SKILL_REGISTRY_TABLES <= set(sa.inspect(store.engine).get_table_names())
