import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from mlflow.store.tracking.dbmodels.models import (
    SqlSkill,
    SqlSkillAlias,
    SqlSkillTag,
    SqlSkillVersion,
    SqlSkillVersionTag,
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


def test_database_cascades_skill_deletion_to_all_children(store):
    _create_skill_with_children(store)
    _create_skill_with_children(store, "survivor")
    skill = store.get_skill("reviewer", organization="acme")
    _assert_skill_row_counts(store, 1)

    with store.ManagedSessionMaker(read_only=False) as session:
        # Core DELETE bypasses ORM relationship cascades and store prechecks.
        session.execute(
            sa.delete(SqlSkill).where(
                SqlSkill.workspace == skill.workspace,
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
