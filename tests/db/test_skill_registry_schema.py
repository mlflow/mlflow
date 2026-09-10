# Cross-dialect (Docker matrix) tests for the RFC-0008 skill registry schema: they run
# against every backend in the MLflow DB test matrix (SQLite, PostgreSQL, MySQL, MSSQL)
# and prove cascade / restrict / uniqueness and latest-resolution behavior at the SQL
# layer rather than through application checks. The store layer does not exist yet, so
# these operate directly on the ORM models.
#
# Several tests here intentionally mirror SQLite-only tests in
# tests/store/tracking/test_skill_registry_dbmodels.py: those give fast feedback in the
# normal test suite, while these re-prove the same guarantees on every engine, where
# constraint enforcement (foreign keys, primary keys) and text sort order can differ
# between databases. Each such test names its counterpart.

from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.store.db.utils import _get_alembic_config
from mlflow.store.tracking.dbmodels.models import (
    SqlAgentPlugin,
    SqlAgentPluginVersion,
    SqlAgentPluginVersionMember,
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


def _seed_assembled_plugin(session, *, organization, name, version, members):
    session.add(SqlAgentPlugin(organization=organization, name=name))
    session.add(
        SqlAgentPluginVersion(
            organization=organization,
            name=name,
            version=version,
            plugin_json={"name": name, "version": version},
            source_type="assembled",
            source="assembled",
        )
    )
    for m_org, m_name, m_ver in members:
        session.add(
            SqlAgentPluginVersionMember(
                plugin_organization=organization,
                plugin_name=name,
                plugin_version=version,
                member_organization=m_org,
                member_name=m_name,
                member_version=m_ver,
            )
        )


def test_db_backend_cascade_delete_skill(store):
    # Cross-dialect twin of test_cascade_delete_skill_removes_children.
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


def test_db_backend_restrict_delete_of_referenced_skill_version(store):
    # Cross-dialect twin of test_restrict_delete_of_skill_version_referenced_by_member.
    with session_scope(store) as session:
        _seed_skill(session, organization="acme", name="member-skill")
        _seed_assembled_plugin(
            session,
            organization="acme",
            name="restrict-plugin",
            version="1.0.0",
            members=[("acme", "member-skill", 1)],
        )
    with session_scope(store, commit=False) as session:
        session.delete(session.get(SqlSkillVersion, ("default", "acme", "member-skill", 1)))
        with pytest.raises(IntegrityError, match=r"(?i)(constraint|duplicate)"):
            session.flush()


def test_db_backend_duplicate_member_name_rejected(store):
    # Cross-dialect twin of test_duplicate_member_name_rejected.
    # Within a single plugin version, the same skill name can't appear more than once --
    # even when the two entries point at different versions of that skill (member_version
    # is deliberately not part of the primary key).
    with session_scope(store) as session:
        _seed_skill(session, organization="acme", name="dup-skill", version=1)
        session.add(
            SqlSkillVersion(
                organization="acme", name="dup-skill", version=2, source_type="git", source="s.git"
            )
        )
    with session_scope(store, commit=False) as session:
        _seed_assembled_plugin(
            session,
            organization="acme",
            name="dup-plugin",
            version="1.0.0",
            members=[("acme", "dup-skill", 1), ("acme", "dup-skill", 2)],
        )
        with pytest.raises(IntegrityError, match=r"(?i)(constraint|duplicate)"):
            session.flush()


# No cross-dialect twins are written for the latest-version resolution tests
# (test_agent_plugin_latest_resolution_prerelease_precedence and test_skill_resolution_*
# in tests/store/tracking/test_skill_registry_dbmodels.py): SemVer ordering,
# active-vs-draft/deprecated preference, and deleted-exclusion are dialect-independent,
# and the sort-key scheme is the MCP registry's (tested in test_semver_utils.py and
# test_mcp_server_registry.py). This file keeps only constraint/migration behavior that
# varies by engine.


_SKILL_REGISTRY_TABLES = frozenset({
    "skills",
    "skill_versions",
    "skill_tags",
    "skill_version_tags",
    "skill_aliases",
    "agent_plugins",
    "agent_plugin_versions",
    "agent_plugin_tags",
    "agent_plugin_version_tags",
    "agent_plugin_aliases",
    "agent_plugin_version_members",
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
