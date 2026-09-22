import pytest

from mlflow.entities import SkillStatus
from mlflow.entities.workspace import Workspace
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import SqlSkill, SqlSkillAlias, SqlSkillVersion
from mlflow.store.tracking.skill_registry.abstract_mixin import SkillRegistryMixin
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

pytestmark = pytest.mark.notrackingurimock


@pytest.mark.parametrize(
    "method_name",
    [
        "get_skill_version_by_alias",
        "get_latest_skill_version",
        "update_skill_version",
        "delete_skill_version",
        "set_skill_alias",
        "delete_skill_alias",
    ],
)
def test_skill_registry_interface_declares_lifecycle_methods(method_name):
    assert callable(getattr(SkillRegistryMixin, method_name, None))


def _seed_skill(store, versions, *, name="reviewer", organization=""):
    with store.ManagedSessionMaker(read_only=False) as session:
        skill = store._with_workspace_field(
            SqlSkill(
                organization=organization,
                name=name,
            )
        )
        session.add(skill)
        for version, status in versions:
            session.add(
                SqlSkillVersion(
                    workspace=skill.workspace,
                    organization=organization,
                    name=name,
                    version=version,
                    status=status.value,
                )
            )


def _add_alias(store, alias, version, *, name="reviewer", organization=""):
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            store._with_workspace_field(
                SqlSkillAlias(
                    organization=organization,
                    name=name,
                    alias=alias,
                    version=version,
                )
            )
        )


def _get_version_row(store, version, *, name="reviewer", organization=""):
    with store.ManagedSessionMaker() as session:
        row = (
            store
            ._get_query(session, SqlSkillVersion)
            .filter(
                SqlSkillVersion.name == name,
                SqlSkillVersion.organization == organization,
                SqlSkillVersion.version == version,
            )
            .one()
        )
        return row.status


def _get_alias_rows(store, *, name="reviewer", organization=""):
    with store.ManagedSessionMaker() as session:
        return (
            store
            ._get_query(session, SqlSkillAlias)
            .filter(
                SqlSkillAlias.name == name,
                SqlSkillAlias.organization == organization,
            )
            .all()
        )


def test_update_skill_version_enforces_lifecycle_transitions(store):
    _seed_skill(store, [(1, SkillStatus.DRAFT)])

    updated = store.update_skill_version("reviewer", 1, status=SkillStatus.ACTIVE)

    assert updated.status == SkillStatus.ACTIVE

    with pytest.raises(MlflowException, match="Invalid status transition") as exc:
        store.update_skill_version("reviewer", 1, status=SkillStatus.DELETED)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert _get_version_row(store, 1) == SkillStatus.ACTIVE.value


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (SkillStatus.DRAFT, SkillStatus.DELETED),
        (SkillStatus.ACTIVE, SkillStatus.DRAFT),
        (SkillStatus.ACTIVE, SkillStatus.DEPRECATED),
        (SkillStatus.DEPRECATED, SkillStatus.ACTIVE),
        (SkillStatus.DEPRECATED, SkillStatus.DELETED),
    ],
)
def test_update_skill_version_allows_valid_lifecycle_transitions(store, current, target):
    _seed_skill(store, [(1, current)])

    updated = store.update_skill_version("reviewer", 1, status=target)

    assert updated.status == target


def test_deleted_skill_version_is_terminal(store):
    _seed_skill(store, [(1, SkillStatus.DELETED)])

    with pytest.raises(MlflowException, match="Invalid status transition"):
        store.update_skill_version("reviewer", 1, status=SkillStatus.DRAFT)


def test_delete_skill_version_soft_deletes_and_removes_aliases(store):
    _seed_skill(store, [(1, SkillStatus.DEPRECATED)])
    _add_alias(store, "production", 1)

    store.delete_skill_version("reviewer", 1)

    assert _get_version_row(store, 1) == SkillStatus.DELETED.value
    assert _get_alias_rows(store) == []


def test_latest_skill_version_prefers_active_then_highest_non_deleted(store):
    _seed_skill(
        store,
        [
            (1, SkillStatus.DEPRECATED),
            (2, SkillStatus.ACTIVE),
            (3, SkillStatus.DRAFT),
        ],
    )

    assert store.get_latest_skill_version("reviewer").version == 2

    store.update_skill_version("reviewer", 2, status=SkillStatus.DEPRECATED)
    store.delete_skill_version("reviewer", 2)

    assert store.get_latest_skill_version("reviewer").version == 3


def test_skill_lifecycle_is_workspace_scoped(store, workspaces_enabled):
    if not workspaces_enabled:
        pytest.skip("Workspace isolation is only applicable when workspaces are enabled")

    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        store._get_workspace_provider_instance().create_workspace(Workspace(name="team-a"))

    with WorkspaceContext("team-a"):
        _seed_skill(store, [(1, SkillStatus.ACTIVE)])
        store.set_skill_alias("reviewer", "team", 1)

    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        with pytest.raises(MlflowException, match="not found"):
            store.get_latest_skill_version("reviewer")
        with pytest.raises(MlflowException, match="not found"):
            store.get_skill_version_by_alias("reviewer", "team")

    with WorkspaceContext("team-a"):
        assert store.get_latest_skill_version("reviewer").version == 1
        assert store.get_skill_version_by_alias("reviewer", "team").version == 1


def test_latest_alias_is_dynamic_and_cannot_be_stored(store):
    _seed_skill(store, [(1, SkillStatus.ACTIVE), (2, SkillStatus.ACTIVE)])

    with pytest.raises(MlflowException, match="reserved") as exc:
        store.set_skill_alias("reviewer", "latest", 1)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert store.get_skill_version_by_alias("reviewer", "latest").version == 2


def test_latest_raises_when_no_non_deleted_version_exists(store):
    _seed_skill(store, [(1, SkillStatus.DELETED)])

    with pytest.raises(MlflowException, match="No resolved latest version") as exc:
        store.get_latest_skill_version("reviewer")

    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


@pytest.mark.parametrize("status", [SkillStatus.DRAFT, SkillStatus.DEPRECATED])
def test_aliases_can_target_non_deleted_versions(store, status):
    _seed_skill(store, [(1, status)])

    store.set_skill_alias("reviewer", "review", 1)

    assert store.get_skill_version_by_alias("reviewer", "review").status == status


def test_aliases_can_be_retargeted(store):
    _seed_skill(store, [(1, SkillStatus.DRAFT), (2, SkillStatus.DEPRECATED)])

    store.set_skill_alias("reviewer", "review", 1)
    store.set_skill_alias("reviewer", "review", 2)

    assert store.get_skill_version_by_alias("reviewer", "review").version == 2


def test_alias_targets_must_exist_and_must_not_be_deleted(store):
    _seed_skill(store, [(1, SkillStatus.DEPRECATED), (2, SkillStatus.DELETED)])

    with pytest.raises(MlflowException, match="not found"):
        store.set_skill_alias("reviewer", "missing", 99)

    with pytest.raises(MlflowException, match="deleted"):
        store.set_skill_alias("reviewer", "deleted", 2)

    _add_alias(store, "stale", 2)
    with pytest.raises(MlflowException, match="missing or deleted"):
        store.get_skill_version_by_alias("reviewer", "stale")


def test_deleting_a_missing_alias_raises(store):
    _seed_skill(store, [(1, SkillStatus.ACTIVE)])

    with pytest.raises(MlflowException, match="Alias 'missing'") as exc:
        store.delete_skill_alias("reviewer", "missing")

    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
