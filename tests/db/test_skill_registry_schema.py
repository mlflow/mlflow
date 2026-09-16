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
#
# NOTE: all tests here deliberately share one database, unlike the SQLite twin file, where
# each test can create and discard its own database cheaply. So keep every test to its own
# data: give skills and plugins names unique to the test that creates them, and filter
# queries by those names rather than relying on a bare `.all()`. Also remember that
# test_db_backend_migration_downgrade_and_reupgrade drops and recreates these tables in
# that same database mid-run.

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
        _seed_assembled_plugin(
            session,
            organization="acme",
            name="raw-cascade-plugin",
            version="1.0.0",
            members=[("acme", "raw-cascade-skill", 1)],
        )

    # Deleting the plugin version must take its members with it, and must NOT touch the
    # skill version those members point at.
    with session_scope(store) as session:
        session.execute(
            sa.delete(SqlAgentPluginVersion).where(
                SqlAgentPluginVersion.workspace == "default",
                SqlAgentPluginVersion.organization == "acme",
                SqlAgentPluginVersion.name == "raw-cascade-plugin",
                SqlAgentPluginVersion.version == "1.0.0",
            )
        )

    with session_scope(store, commit=False) as session:
        assert (
            session
            .query(SqlAgentPluginVersionMember)
            .filter_by(plugin_name="raw-cascade-plugin")
            .count()
            == 0
        )
        assert session.query(SqlSkillVersion).filter_by(name="raw-cascade-skill").count() == 1

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


def test_db_backend_restrict_delete_of_referenced_skill_version(store):
    # Cross-dialect twin of test_restrict_delete_of_skill_version_referenced_by_member.
    # Deletes through SQLAlchemy Core, so it is the FK doing the rejecting.
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
        with pytest.raises(IntegrityError, match=r"(?i)(constraint|duplicate)"):
            session.execute(
                sa.delete(SqlSkillVersion).where(
                    SqlSkillVersion.workspace == "default",
                    SqlSkillVersion.organization == "acme",
                    SqlSkillVersion.name == "member-skill",
                    SqlSkillVersion.version == 1,
                )
            )


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


@pytest.mark.parametrize(
    ("label", "upper", "lower"),
    [
        ("prerelease", "1.0.0-A", "1.0.0-a"),
        # SemVer excludes build metadata from precedence, so the raw `version`
        # string is the only differentiator here.
        ("build", "1.0.0+Build", "1.0.0+build"),
    ],
)
def test_db_backend_version_identity_is_case_sensitive(store, label, upper, lower):
    # SemVer treats `1.0.0-A` and `1.0.0-a`, or `1.0.0+Build` and `1.0.0+build`, as two
    # different plugin versions. But MySQL and SQL Server default to case-insensitive
    # collations and treat them as the same: registering the second violates the primary
    # key, and an exact lookup for one returns the other. The version columns pin a
    # case-sensitive collation on those dialects.
    name = f"case-{label}"
    with session_scope(store) as session:
        session.add(SqlAgentPlugin(organization="acme", name=name))
        for version in (upper, lower):
            session.add(
                SqlAgentPluginVersion(
                    organization="acme",
                    name=name,
                    version=version,
                    plugin_json={"name": name, "version": version},
                    source_type="assembled",
                    source="assembled",
                )
            )

    with session_scope(store, commit=False) as session:
        # Both exist, and each exact lookup returns itself rather than its twin.
        assert session.get(SqlAgentPluginVersion, ("default", "acme", name, upper)).version == (
            upper
        )
        assert session.get(SqlAgentPluginVersion, ("default", "acme", name, lower)).version == (
            lower
        )
        stored = {
            row.version
            for row in session.query(SqlAgentPluginVersion).filter(
                SqlAgentPluginVersion.name == name
            )
        }
        assert stored == {upper, lower}


def test_db_backend_member_version_collation_matches_parent(store):
    # agent_plugin_version_members.plugin_version has to carry the same collation as
    # agent_plugin_versions.version, because MySQL rejects a foreign key whose columns
    # disagree on collation (error 3780). That is already proved before this body runs:
    # the FK is declared by `op.create_table` in the skill-registry migration, applied
    # once when the first test in this file creates the tables (see the header), so a
    # collation mismatch would abort the module during setup rather than fail here.
    #
    # The rows inserted below add the behaviour: given two plugin versions differing only
    # in case, the member row stays attached to the one it was created against.
    with session_scope(store) as session:
        _seed_skill(session, organization="acme", name="members-case-skill")
        _seed_assembled_plugin(
            session,
            organization="acme",
            name="members-case",
            version="2.0.0-A",
            members=[("acme", "members-case-skill", 1)],
        )
        session.add(
            SqlAgentPluginVersion(
                organization="acme",
                name="members-case",
                version="2.0.0-a",
                plugin_json={"name": "members-case", "version": "2.0.0-a"},
                source_type="assembled",
                source="assembled",
            )
        )

    with session_scope(store, commit=False) as session:
        members = (
            session
            .query(SqlAgentPluginVersionMember)
            .filter(SqlAgentPluginVersionMember.plugin_name == "members-case")
            .all()
        )
        assert [(m.plugin_version, m.member_name) for m in members] == [
            ("2.0.0-A", "members-case-skill")
        ]
