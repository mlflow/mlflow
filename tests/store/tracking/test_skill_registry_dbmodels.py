# ORM-level tests for the RFC-0008 skill registry models: ORM mappings,
# ``to_mlflow_entity`` conversion, pure-SQL latest resolution (including the
# withdrawal rule), and database-level
# cascade / restrict / uniqueness constraints against a migrated SQLite database
# (the store layer does not exist yet). Constraint proofs use a foreign-key-
# enforcing session so they assert real DB behavior. These run on SQLite in the normal
# test suite for fast feedback; the same constraint/resolution guarantees are re-proved
# on every engine in the Docker matrix at tests/db/test_skill_registry_schema.py.

from contextlib import contextmanager
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mlflow.entities import (
    GitSource,
    MlflowSource,
    OCISource,
    SkillSourceType,
    SkillStatus,
    ZipSource,
)
from mlflow.store.db.utils import _get_alembic_config
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
        # Standalone MLflow upload: whole tree is the skill, no subpath.
        ("mlflow", "skills/code-review/tok/", None, None, MlflowSource("skills/code-review/tok/")),
        # Member imported from an MLflow-stored package: package tree + subpath.
        (
            "mlflow",
            "agent-plugins/pr/tok/",
            None,
            "code-review",
            MlflowSource("agent-plugins/pr/tok/", "code-review"),
        ),
    ],
)
def test_skill_version_rebuilds_typed_source(store, source_type, source, ref, subpath, expected):
    # to_mlflow_entity rebuilds the typed source object from the flat source columns.
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
    # to_mlflow_entity returns a version's tags as a dict and its aliases as a list of names.
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


# The four tests below check a skill's latest-version resolution logic:
# highest active wins, else highest non-deleted non-active; deleted is excluded.
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
        _add_skill_version(session, version=3, source_type="git", source="s.git", status="draft")
        _add_skill_version(session, version=4, source_type="git", source="s.git", status="deleted")
    with session_scope(store, commit=False) as session:
        ent = SqlSkill.with_resolved_latest(session.query(SqlSkill)).one().to_mlflow_entity()
    # No active version, so the highest non-deleted version wins by number: draft v3 beats
    # deprecated v1, and deleted v4 is excluded even though it is the highest number.
    assert ent.latest_version == 3
    assert ent.status == SkillStatus.DRAFT


def test_skill_resolution_is_none_when_all_versions_deleted(store):
    with session_scope(store) as session:
        _add_skill_version(session, version=1, source_type="git", source="s.git", status="deleted")
        _add_skill_version(session, version=2, source_type="git", source="s.git", status="deleted")
    with session_scope(store, commit=False) as session:
        ent = SqlSkill.with_resolved_latest(session.query(SqlSkill)).one().to_mlflow_entity()
    # All versions deleted -> nothing resolves, but the skill still returns (LEFT JOIN)
    # with latest_version and status both None.
    assert ent.latest_version is None
    assert ent.status is None


def test_skill_resolution_across_many_skills(store):
    # Realistic table: multiple different skills
    with session_scope(store) as session:

        def sv(name, version, status, organization="acme"):
            _add_skill_version(
                session,
                organization=organization,
                name=name,
                version=version,
                source_type="git",
                source="s",
                status=status,
            )

        sv("alpha", 1, "active")
        sv("alpha", 2, "active")
        sv("alpha", 3, "draft")  # active wins over a later draft -> 2
        sv("beta", 1, "deprecated")
        sv("beta", 3, "draft")
        sv("beta", 4, "deleted")  # no active -> highest non-deleted non-active -> 3
        sv("gamma", 1, "deleted")  # all deleted -> None
        sv("alpha", 5, "active", organization="other")  # same name, different org -> 5
    with session_scope(store, commit=False) as session:
        resolved = {}
        for s in SqlSkill.with_resolved_latest(session.query(SqlSkill)).all():
            e = s.to_mlflow_entity()
            resolved[(s.organization, s.name)] = (e.latest_version, e.status)
    assert resolved == {
        ("acme", "alpha"): (2, SkillStatus.ACTIVE),
        ("acme", "beta"): (3, SkillStatus.DRAFT),
        ("acme", "gamma"): (None, None),
        ("other", "alpha"): (5, SkillStatus.ACTIVE),
    }


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
    # Latest-version resolution over a realistic table. SemVer ordering reuses the
    # MCP registry's sort key (mlflow/utils/semver_utils.py); this checks our
    # _version_order_by wires it in and picks the right winner in each case below.
    with session_scope(store) as session:
        made = set()

        def apv(name, version, status, organization="acme"):
            if (organization, name) not in made:
                session.add(SqlAgentPlugin(organization=organization, name=name))
                made.add((organization, name))
            session.add(
                SqlAgentPluginVersion(
                    organization=organization,
                    name=name,
                    version=version,
                    plugin_json={"name": name, "version": version},
                    source_type="assembled",
                    source="assembled",
                    status=status,
                )
            )

        # Numeric prerelease order: alpha.10 > alpha.2 (compared numerically, not as strings).
        apv("aa", "1.0.0-alpha.10", "active")
        apv("aa", "1.0.0-alpha.2", "active")  # -> 1.0.0-alpha.10
        # A full release outranks any prerelease of the same core.
        apv("bb", "1.0.0-rc.1", "active")
        apv("bb", "1.0.0", "active")  # -> 1.0.0
        # A higher core wins even as a prerelease.
        apv("cc", "1.0.0", "active")
        apv("cc", "2.0.0-alpha.1", "active")  # -> 2.0.0-alpha.1
        # No active -> highest non-deleted non-active by SemVer; deleted excluded.
        apv("dd", "1.0.0", "deprecated")
        apv("dd", "1.1.0", "draft")
        apv("dd", "2.0.0", "deleted")  # -> 1.1.0 (draft)
        # All versions deleted -> nothing resolves.
        apv("ee", "1.0.0", "deleted")  # -> None
        # Same name in a different org resolves independently.
        apv("aa", "9.9.9", "active", organization="other")
    with session_scope(store, commit=False) as session:
        resolved = {}
        for p in SqlAgentPlugin.with_resolved_latest(session.query(SqlAgentPlugin)).all():
            e = p.to_mlflow_entity()
            resolved[(p.organization, p.name)] = (e.latest_version, e.status)
    assert resolved == {
        ("acme", "aa"): ("1.0.0-alpha.10", SkillStatus.ACTIVE),
        ("acme", "bb"): ("1.0.0", SkillStatus.ACTIVE),
        ("acme", "cc"): ("2.0.0-alpha.1", SkillStatus.ACTIVE),
        ("acme", "dd"): ("1.1.0", SkillStatus.DRAFT),
        ("acme", "ee"): (None, None),
        ("other", "aa"): ("9.9.9", SkillStatus.ACTIVE),
    }


def test_cascade_delete_skill_removes_children(store):
    # Deletes through the ORM (not raw DB constraints), so `cascade="all, delete-orphan"`
    # removes the children. The database-level proof is
    # test_db_backend_cascade_is_enforced_by_the_database in tests/db.
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
    # As above: the ORM removes the members, not the FK.
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
    # Deletes through SQLAlchemy Core, so it is the FK doing the rejecting.
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
        with pytest.raises(IntegrityError, match=r"(?i)(constraint|duplicate)"):
            session.execute(
                sa.delete(SqlSkillVersion).where(
                    SqlSkillVersion.workspace == "default",
                    SqlSkillVersion.organization == "acme",
                    SqlSkillVersion.name == "code-review",
                    SqlSkillVersion.version == 1,
                )
            )


def test_duplicate_member_name_rejected(store):
    # Within a single plugin version (a plugin bundle), the same skill name can't appear
    # more than once -- even when the two entries point at different versions of that
    # skill (member_version is deliberately not part of the primary key).
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
        # The first member (code-review @ v1) inserts fine on its own.
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
        session.flush()
        # A second member with the same name and different skill version
        # is rejected.
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


def test_duplicate_member_name_rejected_across_organizations(store):
    # A plugin version cannot bundle two skills that share a name even when they are
    # genuinely different skills in different organizations (acme/code-review vs
    # other/code-review). Because when pulling, an assembled plugin writes each
    # member to skills/<member-name>/ on disk, keyed on the bare name. Separate Skills
    # with same name but different org would collide.
    with session_scope(store) as session:
        _add_skill_version(
            session,
            organization="acme",
            name="code-review",
            version=1,
            source_type="git",
            source="a.git",
        )
        _add_skill_version(
            session,
            organization="other",
            name="code-review",
            version=1,
            source_type="git",
            source="o.git",
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
        session.flush()
        session.add(
            SqlAgentPluginVersionMember(
                plugin_organization="acme",
                plugin_name="pr",
                plugin_version="1.0.0",
                member_organization="other",
                member_name="code-review",
                member_version=1,
            )
        )
        with pytest.raises(IntegrityError, match=r"(?i)(constraint|duplicate)"):
            session.flush()


def test_member_name_over_validator_limit_reads_back(store):
    # Name validation happens once, when the user registers the skill -- that is where the
    # 64-char name cap runs. Reading a row back trusts what the database already stored, so
    # to_mlflow_entity must not re-run that cap. A 100-char name (the column allows 128) is a
    # DB-legal value the registration validators would reject, so storing one and projecting
    # it through to_mlflow_entity is our indirect check that the official ORM read method
    # builds the entity without invoking any registration-time validator.
    long_name = "a" * 100
    with session_scope(store) as session:
        _add_skill_version(
            session,
            organization="acme",
            name=long_name,
            version=1,
            source_type="git",
            source="s.git",
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
        session.flush()
        session.add(
            SqlAgentPluginVersionMember(
                plugin_organization="acme",
                plugin_name="pr",
                plugin_version="1.0.0",
                member_organization="acme",
                member_name=long_name,
                member_version=1,
            )
        )
    with session_scope(store, commit=False) as session:
        ent = session.query(SqlAgentPluginVersion).one().to_mlflow_entity()
    assert ent.skills == [f"skills:/@acme/{long_name}/1"]


def test_search_text_round_trip(store):
    # Ensure search_text persists and reads back through the ORM.
    with session_scope(store) as session:
        session.add(
            SqlSkill(
                organization="acme",
                name="code-review",
                search_text="code review lint",
            )
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
                search_text="pr workflow assemble",
            )
        )
    with session_scope(store, commit=False) as session:
        skill = session.get(SqlSkill, ("default", "acme", "code-review"))
        assert skill.search_text == "code review lint"
        plugin_version = session.query(SqlAgentPluginVersion).one()
        assert plugin_version.search_text == "pr workflow assemble"


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


def test_migration_downgrade_and_reupgrade(store, db_uri):
    # Fast SQLite check that the migration downgrades cleanly (the cross-dialect twin
    # is in tests/db/test_skill_registry_schema.py). db_uri is an isolated, per-test
    # copy, so downgrading it here does not affect other tests.
    config = _get_alembic_config(db_uri)
    assert _SKILL_REGISTRY_TABLES <= set(sa.inspect(store.engine).get_table_names())

    # Downgrade removes exactly this migration's tables (FK-safe drop order).
    command.downgrade(config, "b7e2c1a4d9f3")
    assert _SKILL_REGISTRY_TABLES.isdisjoint(set(sa.inspect(store.engine).get_table_names()))

    # Re-upgrade restores them, proving the up/down pair round-trips.
    command.upgrade(config, "e7d1f4b2a9c6")
    assert _SKILL_REGISTRY_TABLES <= set(sa.inspect(store.engine).get_table_names())


# ---------------------------------------------------------------------------
# Withdrawal (RFC-0008)
#
# A plugin version's *members* are the skill versions it bundles. Deleting a
# skill version *withdraws* every plugin version that bundles it: that plugin
# version stops taking part in resolving its parent's latest version and status,
# as if it did not exist.
#
# Its own stored status is never rewritten, because the plugin may belong to
# someone other than whoever deleted the skill version.
#
# Only plugin versions are withdrawn -- skill versions are simply deleted. And
# only a member with `deleted` status withdraws; a `deprecated` member changes nothing.
# ---------------------------------------------------------------------------


def test_deleting_a_bundled_skill_takes_the_plugin_out_of_circulation(store):
    """Deleting a skill version withdraws every plugin version that bundles it. The parent
    plugin keeps resolving as long as one of its versions survives; when all of them are
    deleted or bundle a deleted skill version, it has no latest version at all.

    `pr` survives because its unreleased 2.0.0 draft bundles the newer skill version, so it
    drops back to that draft. `changelog` has nothing left: its 1.0.0 is withdrawn by the
    deleted skill, and its 2.0.0 was already deleted.
    """
    # (plugin, its version, that version's status, which code-review version it bundles)
    plugin_versions = [
        ("pr", "1.0.0", "active", 1),
        ("pr", "2.0.0", "draft", 2),
        ("changelog", "1.0.0", "active", 1),
        ("changelog", "2.0.0", "deleted", 2),
    ]
    with session_scope(store) as session:
        for skill_version in (1, 2):
            _add_skill_version(
                session,
                version=skill_version,
                status="active",
                source_type="git",
                source="pkg.git",
            )
        for name in ("pr", "changelog"):
            session.add(SqlAgentPlugin(organization="acme", name=name))
        for name, version, status, _ in plugin_versions:
            session.add(
                SqlAgentPluginVersion(
                    organization="acme",
                    name=name,
                    version=version,
                    plugin_json={"name": name, "version": version},
                    status=status,
                    source_type="assembled",
                )
            )
        session.flush()
        for name, version, _, skill_version in plugin_versions:
            session.add(
                SqlAgentPluginVersionMember(
                    plugin_organization="acme",
                    plugin_name=name,
                    plugin_version=version,
                    member_organization="acme",
                    member_name="code-review",
                    member_version=skill_version,
                )
            )

    with session_scope(store, commit=False) as session:
        # Both plugins resolve normally while every bundled skill is live. Each one's
        # 1.0.0 outranks its newer 2.0.0: an active version beats a draft, and a deleted
        # version never wins however high its SemVer.
        assert {
            p.name: (p.resolved_latest_version, p.resolved_status)
            for p in SqlAgentPlugin.with_resolved_latest(session.query(SqlAgentPlugin))
        } == {"pr": ("1.0.0", "active"), "changelog": ("1.0.0", "active")}

    # code-review v1 turns out to be compromised, so acme deletes that version.
    with session_scope(store) as session:
        session.get(SqlSkillVersion, ("default", "acme", "code-review", 1)).status = "deleted"

    with session_scope(store, commit=False) as session:
        # Both 1.0.0s are withdrawn. pr falls back to its draft on v2 -- which also shows
        # withdrawal keys on a plugin version's own members, not on any deleted skill in
        # the registry. changelog is out of versions: 1.0.0 withdrawn, 2.0.0 deleted.
        assert {
            p.name: (p.resolved_latest_version, p.resolved_status)
            for p in SqlAgentPlugin.with_resolved_latest(session.query(SqlAgentPlugin))
        } == {"pr": ("2.0.0", "draft"), "changelog": (None, None)}


def test_deleting_a_skill_does_not_write_to_plugin_version_rows(store):
    """
    `beta/dev-tools` 2.1.0 and `gamma/ci-suite` 1.4.0 both bundle `acme/code-review` v3.
    `gamma/ci-suite` also keeps an older 1.3.0, which bundles the `deprecated` v2.

    Deleting `acme/code-review` v3 then:
    - writes one `skill_versions` row and nothing at all in `agent_plugin_versions` or
      `agent_plugin_version_members`
    - withdraws the two plugin versions that bundle it, 2.1.0 and 1.4.0, so neither
      takes part in resolution any more
    - leaves `beta/dev-tools` with no latest version, while `gamma/ci-suite` falls back
      to 1.3.0, whose member is only `deprecated` and so does not withdraw it
    """
    # (organization, plugin, its version, that version's status, acme/code-review version bundled)
    plugin_versions = [
        ("beta", "dev-tools", "2.1.0", "active", 3),
        ("gamma", "ci-suite", "1.3.0", "deprecated", 2),
        ("gamma", "ci-suite", "1.4.0", "active", 3),
    ]
    with session_scope(store) as session:
        _add_skill_version(session, version=2, status="deprecated")
        _add_skill_version(session, version=3, status="active")
        for organization, name in [("beta", "dev-tools"), ("gamma", "ci-suite")]:
            session.add(SqlAgentPlugin(organization=organization, name=name))
        for organization, name, version, status, _ in plugin_versions:
            session.add(
                SqlAgentPluginVersion(
                    organization=organization,
                    name=name,
                    version=version,
                    plugin_json={"name": name, "version": version},
                    status=status,
                    source_type="assembled",
                )
            )
        session.flush()
        for organization, name, version, _, member_version in plugin_versions:
            session.add(
                SqlAgentPluginVersionMember(
                    plugin_organization=organization,
                    plugin_name=name,
                    plugin_version=version,
                    member_organization="acme",
                    member_name="code-review",
                    member_version=member_version,
                )
            )

    with session_scope(store, commit=False) as session:
        # Every bundled skill is live, so both plugins resolve to their newest version.
        assert {
            p.organization: (p.resolved_latest_version, p.resolved_status)
            for p in SqlAgentPlugin.with_resolved_latest(session.query(SqlAgentPlugin))
        } == {"beta": ("2.1.0", "active"), "gamma": ("1.4.0", "active")}

    # A user deletes acme/code-review v3. That is one UPDATE, on one skill_versions row.
    with session_scope(store) as session:
        session.get(SqlSkillVersion, ("default", "acme", "code-review", 3)).status = "deleted"

    with session_scope(store, commit=False) as session:
        # Every agent_plugin_versions.status is untouched: the delete wrote only to
        # skill_versions, so these still read exactly what they read above.
        assert {
            (v.organization, v.version): v.status for v in session.query(SqlAgentPluginVersion)
        } == {
            ("beta", "2.1.0"): "active",
            ("gamma", "1.3.0"): "deprecated",
            ("gamma", "1.4.0"): "active",
        }
        # agent_plugin_version_members rows are a historical record, so none is removed.
        assert session.query(SqlAgentPluginVersionMember).count() == 3

        # 2.1.0 and 1.4.0 are withdrawn now, so beta/dev-tools has no latest version,
        # and gamma/ci-suite resolves to 1.3.0, whose member is only `deprecated`.
        assert {
            p.organization: (p.resolved_latest_version, p.resolved_status)
            for p in SqlAgentPlugin.with_resolved_latest(session.query(SqlAgentPlugin))
        } == {"beta": (None, None), "gamma": ("1.3.0", "deprecated")}


def test_one_deleted_skill_withdraws_a_plugin_version_even_if_its_other_skills_are_fine(store):
    """
    `p` 1.0.0 bundles three skills. Deleting just one of them withdraws 1.0.0, even
    though the other two are fine, and `p` falls back to its older 0.9.0. That a
    `deprecated` member does not withdraw is proved by the first assertion, where
    1.0.0 still wins with `sb` deprecated.
    """
    with session_scope(store) as session:
        for name, status in [
            ("sa", "active"),
            ("sb", "deprecated"),
            ("sc", "active"),
            ("sd", "active"),
        ]:
            _add_skill_version(
                session,
                name=name,
                version=1,
                status=status,
                source_type="git",
                source="pkg.git",
            )
        session.add(SqlAgentPlugin(organization="acme", name="p"))
        # plugin version -> the skills it bundles (every skill here is at v1)
        members_by_version = {"0.9.0": ["sd"], "1.0.0": ["sa", "sb", "sc"]}
        for version in members_by_version:
            session.add(
                SqlAgentPluginVersion(
                    organization="acme",
                    name="p",
                    version=version,
                    plugin_json={"name": "p", "version": version},
                    status="active",
                    source_type="assembled",
                )
            )
        session.flush()
        for version, members in members_by_version.items():
            for member_name in members:
                session.add(
                    SqlAgentPluginVersionMember(
                        plugin_organization="acme",
                        plugin_name="p",
                        plugin_version=version,
                        member_organization="acme",
                        member_name=member_name,
                        member_version=1,
                    )
                )

    with session_scope(store, commit=False) as session:
        # Nothing is deleted yet, so the newer 1.0.0 wins -- sb being deprecated is fine.
        plugin = SqlAgentPlugin.with_resolved_latest(session.query(SqlAgentPlugin)).one()
        assert (plugin.resolved_latest_version, plugin.resolved_status) == ("1.0.0", "active")

    # One of 1.0.0's three skills is deleted. sb and sc are untouched.
    with session_scope(store) as session:
        session.get(SqlSkillVersion, ("default", "acme", "sa", 1)).status = "deleted"

    with session_scope(store, commit=False) as session:
        plugin = SqlAgentPlugin.with_resolved_latest(session.query(SqlAgentPlugin)).one()
        assert (plugin.resolved_latest_version, plugin.resolved_status) == ("0.9.0", "active")

        # Resolution only reads. Both plugin versions and all four member rows are still here.
        assert session.query(SqlAgentPluginVersion).count() == 2
        assert session.query(SqlAgentPluginVersionMember).count() == 4
        assert session.query(SqlSkillVersion).count() == 4
