# ORM-level tests for the RFC-0008 skill registry models: ORM mappings,
# ``to_mlflow_entity`` conversion, pure-SQL latest resolution, and database-level
# cascade / restrict / uniqueness constraints against a migrated SQLite database
# (the store layer does not exist yet). Constraint proofs use a foreign-key-
# enforcing session so they assert real DB behavior. Cross-dialect coverage lives
# in tests/db/test_skill_registry_schema.py.

from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mlflow.entities import (
    GitSource,
    OCISource,
    SkillSourceType,
    SkillStatus,
    ZipSource,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlAgentPlugin,
    SqlAgentPluginAlias,
    SqlAgentPluginVersion,
    SqlAgentPluginVersionMember,
    SqlAgentPluginVersionTag,
    SqlSkill,
    SqlSkillAlias,
    SqlSkillTag,
    SqlSkillVersion,
    SqlSkillVersionTag,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore


@pytest.fixture
def store(tmp_path: Path, db_uri: str):
    artifact_uri = tmp_path / "artifacts"
    artifact_uri.mkdir()
    return SqlAlchemyStore(db_uri, artifact_uri.as_uri())


@contextmanager
def session_scope(store, *, commit=True):
    """A foreign-key-enforcing session bound to the store engine."""
    with Session(store.engine) as session:
        if store.engine.dialect.name == "sqlite":
            session.execute(sa.text("PRAGMA foreign_keys = ON"))
        yield session
        if commit:
            session.commit()


def _add_skill_version(session, *, organization="acme", name="code-review", version=1, **kwargs):
    if session.get(SqlSkill, (kwargs.get("workspace", "default"), organization, name)) is None:
        session.add(
            SqlSkill(
                workspace=kwargs.get("workspace", "default"),
                organization=organization,
                name=name,
            )
        )
    sv = SqlSkillVersion(organization=organization, name=name, version=version, **kwargs)
    session.add(sv)
    return sv


@pytest.mark.parametrize(
    ("source_type", "source", "ref", "subpath", "expected"),
    [
        (
            "git",
            "https://github.com/acme/s.git",
            "v1",
            "sub",
            GitSource("https://github.com/acme/s.git", "v1", "sub"),
        ),
        ("oci", "ghcr.io/acme/s:v1", None, "sub", OCISource("ghcr.io/acme/s:v1", "sub")),
        ("zip", "https://acme/s.zip", None, "sub", ZipSource("https://acme/s.zip", "sub")),
        ("mlflow", "skills/code-review/tok/", None, None, "skills/code-review/tok/"),
    ],
)
def test_skill_version_rebuilds_typed_source(store, source_type, source, ref, subpath, expected):
    with session_scope(store) as session:
        _add_skill_version(
            session, source_type=source_type, source=source, ref=ref, subpath=subpath, digest="d"
        )
    with session_scope(store, commit=False) as session:
        ent = session.query(SqlSkillVersion).one().to_mlflow_entity()
    assert ent.source == expected
    assert ent.source_type == SkillSourceType(source_type)
    assert ent.digest == "d"


def test_skill_version_projects_tags_and_aliases(store):
    with session_scope(store) as session:
        _add_skill_version(session, version=1, source_type="git", source="s.git")
        session.add(
            SqlSkillVersionTag(
                organization="acme", name="code-review", version=1, key="k", value="v"
            )
        )
        session.add(SqlSkillAlias(organization="acme", name="code-review", alias="prod", version=1))
        session.add(
            SqlSkillAlias(organization="acme", name="code-review", alias="staging", version=1)
        )
    with session_scope(store, commit=False) as session:
        ent = session.query(SqlSkillVersion).one().to_mlflow_entity()
    assert ent.tags == {"k": "v"}
    assert sorted(ent.aliases) == ["prod", "staging"]


def test_skill_resolution_and_entity(store):
    with session_scope(store) as session:
        _add_skill_version(session, version=1, source_type="git", source="s.git", status="active")
        _add_skill_version(session, version=2, source_type="git", source="s.git", status="active")
        _add_skill_version(session, version=3, source_type="git", source="s.git", status="draft")
        session.add(SqlSkillTag(organization="acme", name="code-review", key="team", value="ml"))
        session.add(SqlSkillAlias(organization="acme", name="code-review", alias="prod", version=2))
    with session_scope(store, commit=False) as session:
        skill = SqlSkill.with_resolved_latest(session.query(SqlSkill)).one()
        ent = skill.to_mlflow_entity()
    # Highest active version wins over a later draft.
    assert ent.latest_version == 2
    assert ent.status == SkillStatus.ACTIVE
    assert ent.tags == {"team": "ml"}
    assert ent.aliases == {"prod": 2}


def test_skill_resolution_falls_back_to_non_active_and_excludes_deleted(store):
    with session_scope(store) as session:
        _add_skill_version(
            session, version=1, source_type="git", source="s.git", status="deprecated"
        )
        _add_skill_version(session, version=2, source_type="git", source="s.git", status="deleted")
    with session_scope(store, commit=False) as session:
        ent = SqlSkill.with_resolved_latest(session.query(SqlSkill)).one().to_mlflow_entity()
    # No active version: highest non-deleted (deprecated v1), deleted v2 excluded.
    assert ent.latest_version == 1
    assert ent.status == SkillStatus.DEPRECATED


@pytest.mark.parametrize(
    "icons",
    [[{"src": "https://acme/icon.png", "sizes": "48x48", "mimeType": "image/png"}], None],
)
def test_skill_and_plugin_icons_round_trip(store, icons):
    # icons is mutable presentation metadata stored as-is and returned exactly as stored
    # (null when unset), with no payload fallback (unlike the MCP registry).
    with session_scope(store) as session:
        session.add(SqlSkill(organization="acme", name="code-review", icons=icons))
        session.add(SqlAgentPlugin(organization="acme", name="pr", icons=icons))
    with session_scope(store, commit=False) as session:
        skill = session.query(SqlSkill).one().to_mlflow_entity()
        plugin = session.query(SqlAgentPlugin).one().to_mlflow_entity()
    assert skill.icons == icons
    assert plugin.icons == icons


def test_agent_plugin_version_materializes_semver_components(store):
    with session_scope(store) as session:
        session.add(SqlAgentPlugin(organization="acme", name="pr"))
        session.add(
            SqlAgentPluginVersion(
                organization="acme",
                name="pr",
                version="1.2.3-beta.1",
                plugin_json={"name": "pr", "version": "1.2.3-beta.1"},
                source_type="assembled",
                source="assembled",
            )
        )
    with session_scope(store, commit=False) as session:
        row = session.query(SqlAgentPluginVersion).one()
    assert (row.version_major, row.version_minor, row.version_patch) == (1, 2, 3)
    assert row.version_prerelease_sort_key  # non-empty encoded key


def test_agent_plugin_version_entity_rebuilds_members_and_json(store):
    with session_scope(store) as session:
        _add_skill_version(
            session, name="code-review", version=1, source_type="git", source="s.git"
        )
        _add_skill_version(session, name="lint", version=2, source_type="git", source="l.git")
        session.add(SqlAgentPlugin(organization="acme", name="pr"))
        session.add(
            SqlAgentPluginVersion(
                organization="acme",
                name="pr",
                version="1.0.0",
                plugin_json={"name": "pr", "version": "1.0.0", "description": "d"},
                source_type="git",
                source="pkg.git",
                ref="v1",
            )
        )
        session.add(
            SqlAgentPluginVersionTag(
                organization="acme", name="pr", version="1.0.0", key="k", value="v"
            )
        )
        session.add(
            SqlAgentPluginAlias(organization="acme", name="pr", alias="prod", version="1.0.0")
        )
        for m_name, m_ver in (("code-review", 1), ("lint", 2)):
            session.add(
                SqlAgentPluginVersionMember(
                    plugin_organization="acme",
                    plugin_name="pr",
                    plugin_version="1.0.0",
                    member_organization="acme",
                    member_name=m_name,
                    member_version=m_ver,
                )
            )
    with session_scope(store, commit=False) as session:
        ent = session.query(SqlAgentPluginVersion).one().to_mlflow_entity()
    assert ent.plugin_json == {"name": "pr", "version": "1.0.0", "description": "d"}
    assert ent.source == GitSource("pkg.git", "v1", None)
    assert ent.tags == {"k": "v"}
    assert ent.aliases == ["prod"]
    assert sorted(ent.skills) == ["skills:/@acme/code-review/1", "skills:/@acme/lint/2"]


def test_agent_plugin_latest_resolution_prerelease_precedence(store):
    with session_scope(store) as session:
        session.add(SqlAgentPlugin(organization="acme", name="pr"))
        for v in ("1.0.0-alpha.2", "1.0.0-alpha.10", "1.0.0-beta.1", "1.0.0"):
            session.add(
                SqlAgentPluginVersion(
                    organization="acme",
                    name="pr",
                    version=v,
                    plugin_json={"name": "pr", "version": v},
                    source_type="assembled",
                    source="assembled",
                )
            )
    with session_scope(store, commit=False) as session:
        ent = (
            SqlAgentPlugin
            .with_resolved_latest(session.query(SqlAgentPlugin))
            .one()
            .to_mlflow_entity()
        )
    # Full release outranks all prereleases.
    assert ent.latest_version == "1.0.0"
    assert ent.status == SkillStatus.ACTIVE


def test_cascade_delete_skill_removes_children(store):
    with session_scope(store) as session:
        _add_skill_version(session, version=1, source_type="git", source="s.git")
        session.add(SqlSkillTag(organization="acme", name="code-review", key="k", value="v"))
        session.add(
            SqlSkillVersionTag(
                organization="acme", name="code-review", version=1, key="k", value="v"
            )
        )
        session.add(SqlSkillAlias(organization="acme", name="code-review", alias="prod", version=1))
    with session_scope(store) as session:
        session.delete(session.get(SqlSkill, ("default", "acme", "code-review")))
    with session_scope(store, commit=False) as session:
        assert session.query(SqlSkillVersion).count() == 0
        assert session.query(SqlSkillTag).count() == 0
        assert session.query(SqlSkillVersionTag).count() == 0
        assert session.query(SqlSkillAlias).count() == 0


def test_cascade_delete_plugin_version_removes_members(store):
    with session_scope(store) as session:
        _add_skill_version(session, version=1, source_type="git", source="s.git")
        session.add(SqlAgentPlugin(organization="acme", name="pr"))
        session.add(
            SqlAgentPluginVersion(
                organization="acme",
                name="pr",
                version="1.0.0",
                plugin_json={"name": "pr", "version": "1.0.0"},
                source_type="assembled",
                source="assembled",
            )
        )
        session.add(
            SqlAgentPluginVersionMember(
                plugin_organization="acme",
                plugin_name="pr",
                plugin_version="1.0.0",
                member_organization="acme",
                member_name="code-review",
                member_version=1,
            )
        )
    with session_scope(store) as session:
        session.delete(session.get(SqlAgentPluginVersion, ("default", "acme", "pr", "1.0.0")))
    with session_scope(store, commit=False) as session:
        assert session.query(SqlAgentPluginVersionMember).count() == 0
        # The member skill version is untouched by the plugin-version delete.
        assert session.query(SqlSkillVersion).count() == 1


def test_restrict_delete_of_skill_version_referenced_by_member(store):
    with session_scope(store) as session:
        _add_skill_version(session, version=1, source_type="git", source="s.git")
        session.add(SqlAgentPlugin(organization="acme", name="pr"))
        session.add(
            SqlAgentPluginVersion(
                organization="acme",
                name="pr",
                version="1.0.0",
                plugin_json={"name": "pr", "version": "1.0.0"},
                source_type="assembled",
                source="assembled",
            )
        )
        session.add(
            SqlAgentPluginVersionMember(
                plugin_organization="acme",
                plugin_name="pr",
                plugin_version="1.0.0",
                member_organization="acme",
                member_name="code-review",
                member_version=1,
            )
        )
    with session_scope(store, commit=False) as session:
        session.delete(session.get(SqlSkillVersion, ("default", "acme", "code-review", 1)))
        with pytest.raises(IntegrityError, match=r"(?i)(constraint|duplicate)"):
            session.flush()


def test_duplicate_member_name_rejected(store):
    with session_scope(store) as session:
        _add_skill_version(
            session, name="code-review", version=1, source_type="git", source="s.git"
        )
        _add_skill_version(
            session, name="code-review", version=2, source_type="git", source="s.git"
        )
        session.add(SqlAgentPlugin(organization="acme", name="pr"))
        session.add(
            SqlAgentPluginVersion(
                organization="acme",
                name="pr",
                version="1.0.0",
                plugin_json={"name": "pr", "version": "1.0.0"},
                source_type="assembled",
                source="assembled",
            )
        )
    with session_scope(store, commit=False) as session:
        # Same member_name twice within one plugin version violates the PK,
        # regardless of differing member_version.
        session.add(
            SqlAgentPluginVersionMember(
                plugin_organization="acme",
                plugin_name="pr",
                plugin_version="1.0.0",
                member_organization="acme",
                member_name="code-review",
                member_version=1,
            )
        )
        session.add(
            SqlAgentPluginVersionMember(
                plugin_organization="acme",
                plugin_name="pr",
                plugin_version="1.0.0",
                member_organization="acme",
                member_name="code-review",
                member_version=2,
            )
        )
        with pytest.raises(IntegrityError, match=r"(?i)(constraint|duplicate)"):
            session.flush()


def test_workspace_isolation_same_org_and_name(store):
    with session_scope(store) as session:
        _add_skill_version(session, workspace="alpha", version=1, source_type="git", source="a.git")
        _add_skill_version(session, workspace="beta", version=1, source_type="git", source="b.git")
    with session_scope(store, commit=False) as session:
        assert session.query(SqlSkill).count() == 2
        alpha = session.query(SqlSkillVersion).filter_by(workspace="alpha").one()
        assert alpha.to_mlflow_entity().source == GitSource("a.git", None, None)
        assert alpha.to_mlflow_entity().workspace == "alpha"
