# Database-level constraint tests for the RFC-0008 skill registry schema. Runs
# against every backend in the MLflow DB test matrix (SQLite, PostgreSQL, MySQL,
# MSSQL) and proves cascade / restrict / uniqueness behavior at the SQL layer
# rather than through application checks. The store layer does not exist yet, so
# these operate directly on the ORM models.

from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mlflow.environment_variables import MLFLOW_TRACKING_URI
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


def test_db_backend_agent_plugin_latest_prefers_release_over_prerelease(store):
    with session_scope(store) as session:
        session.add(SqlAgentPlugin(organization="acme", name="latest-plugin"))
        for v in ("1.0.0-alpha.2", "1.0.0-alpha.10", "1.0.0-beta.1", "1.0.0"):
            session.add(
                SqlAgentPluginVersion(
                    organization="acme",
                    name="latest-plugin",
                    version=v,
                    plugin_json={"name": "latest-plugin", "version": v},
                    source_type="assembled",
                    source="assembled",
                )
            )
    with session_scope(store, commit=False) as session:
        plugin = (
            SqlAgentPlugin
            .with_resolved_latest(session.query(SqlAgentPlugin))
            .filter_by(organization="acme", name="latest-plugin")
            .one()
        )
        assert plugin.to_mlflow_entity().latest_version == "1.0.0"


def test_db_backend_skill_latest_prefers_active(store):
    with session_scope(store) as session:
        session.add(SqlSkill(organization="acme", name="latest-skill"))
        session.add(
            SqlSkillVersion(
                organization="acme",
                name="latest-skill",
                version=1,
                source_type="git",
                source="s.git",
                status="active",
            )
        )
        session.add(
            SqlSkillVersion(
                organization="acme",
                name="latest-skill",
                version=2,
                source_type="git",
                source="s.git",
                status="draft",
            )
        )
    with session_scope(store, commit=False) as session:
        skill = (
            SqlSkill
            .with_resolved_latest(session.query(SqlSkill))
            .filter_by(organization="acme", name="latest-skill")
            .one()
        )
        assert skill.to_mlflow_entity().latest_version == 1
