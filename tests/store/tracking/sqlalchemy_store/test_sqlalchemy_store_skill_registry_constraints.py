import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from mlflow.store.tracking.dbmodels.models import (
    SqlAgentPluginVersion,
    SqlAgentPluginVersionMember,
    SqlSkill,
    SqlSkillAlias,
    SqlSkillTag,
    SqlSkillVersion,
    SqlSkillVersionTag,
)

from tests.store.tracking.sqlalchemy_store.test_sqlalchemy_store_skill_registry_delete import (
    _add_plugin_version,
    _workspace,
)

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture(autouse=True)
def assert_foreign_keys_enabled(store):
    if store.engine.dialect.name == "sqlite":
        with store.ManagedSessionMaker() as session:
            assert session.scalar(sa.text("PRAGMA foreign_keys")) == 1


def _create_skill_with_children(store, name="reviewer"):
    store.create_skill_version(name, organization="acme")
    store.set_skill_tag(name, "team", "platform", organization="acme")
    store.set_skill_version_tag(name, 1, "release", "stable", organization="acme")
    store.set_skill_alias(name, "production", 1, organization="acme")


def _assert_skill_row_counts(store, expected, name="reviewer"):
    with store.ManagedSessionMaker() as session:
        for model in (SqlSkill, SqlSkillVersion, SqlSkillTag, SqlSkillVersionTag, SqlSkillAlias):
            assert (
                store._get_query(session, model).filter_by(organization="acme", name=name).count()
                == expected
            )


def _membership(store, **overrides):
    fields = {
        "plugin_workspace": _workspace(store),
        "plugin_organization": "acme",
        "plugin_name": "toolkit",
        "plugin_version": "1.0.0",
        "member_organization": "acme",
        "member_name": "reviewer",
        "member_version": 1,
    }
    return SqlAgentPluginVersionMember(**(fields | overrides))


def test_database_cascades_skill_deletion_to_all_children(store):
    _create_skill_with_children(store)
    _create_skill_with_children(store, "survivor")
    _assert_skill_row_counts(store, 1)

    with store.ManagedSessionMaker(read_only=False) as session:
        # Core DELETE bypasses ORM relationship cascades and store prechecks.
        session.execute(
            sa.delete(SqlSkill).where(
                SqlSkill.workspace == _workspace(store),
                SqlSkill.organization == "acme",
                SqlSkill.name == "reviewer",
            )
        )

    _assert_skill_row_counts(store, 0)
    _assert_skill_row_counts(store, 1, "survivor")


@pytest.mark.parametrize(
    ("model", "fields"),
    [
        (SqlSkillVersion, {"name": "missing", "version": 1}),
        (SqlSkillTag, {"name": "missing", "key": "team", "value": "platform"}),
        (SqlSkillAlias, {"name": "missing", "alias": "production", "version": 1}),
        (
            SqlSkillVersionTag,
            {"name": "reviewer", "version": 99, "key": "release", "value": "stable"},
        ),
    ],
)
def test_database_rejects_orphan_skill_children(store, model, fields):
    _create_skill_with_children(store)
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(store._with_workspace_field(model(organization="acme", **fields)))
        with pytest.raises(IntegrityError, match=r"(?i)(foreign key|constraint)"):
            session.flush()
        session.rollback()

    _assert_skill_row_counts(store, 1)
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, model).count() == 1


@pytest.mark.parametrize("model", [SqlSkill, SqlSkillVersion])
@pytest.mark.parametrize("plugin_status", ["active", "deleted"])
def test_database_restricts_deletion_while_membership_exists(store, model, plugin_status):
    _create_skill_with_children(store)
    _add_plugin_version(store, [("reviewer", 1)], status=plugin_status)
    with store.ManagedSessionMaker(read_only=False) as session:
        # Even stale references block raw SQL; the store must purge them first.
        with pytest.raises(IntegrityError, match=r"(?i)(foreign key|constraint)"):
            session.execute(
                sa.delete(model).where(
                    model.workspace == _workspace(store),
                    model.organization == "acme",
                    model.name == "reviewer",
                )
            )
        session.rollback()

    _assert_skill_row_counts(store, 1)
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlAgentPluginVersionMember).count() == 1


def test_database_plugin_version_cascade_preserves_member_skill(store):
    _create_skill_with_children(store)
    _add_plugin_version(store, [("reviewer", 1)])
    with store.ManagedSessionMaker(read_only=False) as session:
        session.execute(
            sa.delete(SqlAgentPluginVersion).where(
                SqlAgentPluginVersion.workspace == _workspace(store),
                SqlAgentPluginVersion.organization == "acme",
                SqlAgentPluginVersion.name == "toolkit",
            )
        )

    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlAgentPluginVersion).count() == 0
        assert session.query(SqlAgentPluginVersionMember).count() == 0
    _assert_skill_row_counts(store, 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"plugin_name": "missing"},
        {"plugin_version": "9.0.0"},
        {"member_name": "missing"},
        {"member_version": 99},
        {"member_organization": "other"},
    ],
)
def test_database_rejects_invalid_membership_references(store, overrides):
    _create_skill_with_children(store)
    _add_plugin_version(store, [])
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(_membership(store, **overrides))
        with pytest.raises(IntegrityError, match=r"(?i)(foreign key|constraint)"):
            session.flush()
        session.rollback()

    _assert_skill_row_counts(store, 1)
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlAgentPluginVersionMember).count() == 0


@pytest.mark.parametrize("organization", ["acme", "other"])
def test_database_rejects_duplicate_member_names(store, organization):
    store.create_skill_version("reviewer", organization="acme")
    other = store.create_skill_version("reviewer", organization=organization)
    _add_plugin_version(store, [("reviewer", 1)])
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            _membership(store, member_organization=organization, member_version=other.version)
        )
        with pytest.raises(IntegrityError, match=r"(?i)(unique|duplicate|primary key)"):
            session.flush()
        session.rollback()

    with store.ManagedSessionMaker() as session:
        member = session.query(SqlAgentPluginVersionMember).one()
        assert (member.member_organization, member.member_version) == ("acme", 1)
    assert store.get_skill_version("reviewer", other.version, organization=organization) == other


def test_database_rejects_membership_to_skill_in_another_workspace(store):
    _add_plugin_version(store, [])
    # Seed a foreign workspace directly to exercise the schema in both store modes.
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(SqlSkill(workspace="other", organization="acme", name="reviewer"))
        session.flush()
        session.add(
            SqlSkillVersion(
                workspace="other",
                organization="acme",
                name="reviewer",
                version=1,
            )
        )
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(_membership(store))
        with pytest.raises(IntegrityError, match=r"(?i)(foreign key|constraint)"):
            session.flush()
        session.rollback()

    with store.ManagedSessionMaker() as session:
        assert session.query(SqlAgentPluginVersionMember).count() == 0
        assert session.query(SqlSkillVersion).filter_by(workspace="other").count() == 1
