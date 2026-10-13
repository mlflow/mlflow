import threading
from unittest import mock

import pytest

from mlflow.entities.skill import SkillStatus
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import (
    SqlSkillVersion,
)
from mlflow.store.tracking.skill_registry.artifact_paths import (
    new_skill_upload_path,
    owned_skill_upload_path,
    to_artifact_uri,
)
from mlflow.utils.workspace_context import WorkspaceContext

pytestmark = pytest.mark.notrackingurimock

_EXTERNAL_TREE = "mlflow-artifacts:/external-content/@acme/toolkit/0123456789abcdef0123456789abcdef"


def _upload(store, name="reviewer", organization="acme"):
    """Create a version the way the upload flow does and return (version, owned path)."""
    path = new_skill_upload_path(name, organization)
    version = store.create_skill_version(
        name, organization=organization, source_type="mlflow", source=to_artifact_uri(path)
    )
    return version, path


def _version_rows(store, name="reviewer"):
    with store.ManagedSessionMaker() as session:
        return session.query(SqlSkillVersion).filter(SqlSkillVersion.name == name).count()


def test_delete_skill_removes_parent_versions_and_returns_owned_paths(store):
    _, first = _upload(store)
    _, second = _upload(store)
    store.create_skill_version(
        "reviewer", organization="acme", source_type="git", source="https://example.com/r.git"
    )

    owned = store.delete_skill_and_collect_artifacts("reviewer", organization="acme")

    assert sorted(owned) == sorted([first, second])
    assert _version_rows(store) == 0
    with pytest.raises(MlflowException, match="not found"):
        store.get_skill("reviewer", organization="acme")


def test_delete_skill_never_returns_a_referenced_tree(store):
    # Deleting the last reference to an external tree must not schedule it for cleanup.
    store.create_skill_version(
        "reviewer",
        organization="acme",
        source_type="mlflow",
        source=_EXTERNAL_TREE,
        subpath="skills/reviewer",
    )
    _, owned_path = _upload(store)

    owned = store.delete_skill_and_collect_artifacts("reviewer", organization="acme")

    assert owned == [owned_path]


def test_delete_skill_returns_none_and_matches_collecting_variant(store):
    _upload(store)
    assert store.delete_skill("reviewer", organization="acme") is None
    assert _version_rows(store) == 0


def test_delete_skill_not_found(store):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.delete_skill_and_collect_artifacts("reviewer", organization="acme")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_delete_skill_restarts_version_allocation(store):
    for _ in range(3):
        _upload(store)
    store.delete_skill("reviewer", organization="acme")

    recreated, path = _upload(store)

    assert recreated.version == 1
    assert store.delete_skill_and_collect_artifacts("reviewer", organization="acme") == [path]


def test_delete_skill_includes_soft_deleted_versions(store):
    # A soft-deleted version keeps its row and its content until the parent is hard-deleted.
    _, path = _upload(store)
    with store.ManagedSessionMaker(read_only=False) as session:
        session.query(SqlSkillVersion).filter(SqlSkillVersion.name == "reviewer").update({
            "status": SkillStatus.DELETED.value
        })

    assert store.delete_skill_and_collect_artifacts("reviewer", organization="acme") == [path]


def test_delete_skill_holds_out_a_concurrent_version_until_it_commits(store):
    # Runs against every backend in the database matrix: the lock taken before the version
    # snapshot must be strong enough to block the foreign-key lock a concurrent version insert
    # takes on the parent, or that version would be cascaded away with its artifact path never
    # captured. The insert waits, then recreates the skill from version 1 after the delete.
    _, path = _upload(store)
    committed = threading.Event()
    outcome = {}

    def register():
        outcome["version"] = store.create_skill_version(
            "reviewer", organization="acme", source_type="git", source="https://h/r.git"
        )
        committed.set()

    thread = threading.Thread(target=register, name="concurrent-registration")

    def classify_in_the_window(**kwargs):
        thread.start()
        assert not committed.wait(timeout=2)
        return owned_skill_upload_path(**kwargs)

    with mock.patch(
        "mlflow.store.tracking.skill_registry.sqlalchemy_mixin.owned_skill_upload_path",
        side_effect=classify_in_the_window,
    ) as classify:
        owned = store.delete_skill_and_collect_artifacts("reviewer", organization="acme")
    classify.assert_called_once()
    thread.join(timeout=30)
    assert committed.is_set()
    assert owned == [path]
    assert outcome["version"].version == 1
    assert _version_rows(store) == 1


def test_delete_skill_leaves_other_identities_alone(store):
    _upload(store, name="reviewer", organization="acme")
    _, other_org_path = _upload(store, name="reviewer", organization="example")
    _, other_name_path = _upload(store, name="linter", organization="acme")

    store.delete_skill("reviewer", organization="acme")

    assert store.get_skill_version("reviewer", 1, organization="example").version == 1
    assert store.get_skill_version("linter", 1, organization="acme").version == 1
    assert store.delete_skill_and_collect_artifacts("reviewer", "example") == [other_org_path]
    assert store.delete_skill_and_collect_artifacts("linter", "acme") == [other_name_path]


def test_delete_skill_is_workspace_scoped(store, workspaces_enabled):
    if not workspaces_enabled:
        pytest.skip("workspace isolation only applies when workspaces are enabled")
    with WorkspaceContext("team-a"):
        _, team_a_path = _upload(store)
    with WorkspaceContext("team-b"):
        # team-b cannot see team-a's skill, so there is nothing for it to delete.
        with pytest.raises(MlflowException, match="not found"):
            store.delete_skill("reviewer", organization="acme")
        # Deleting its own skill of the same name leaves team-a's untouched.
        _upload(store)
        store.delete_skill("reviewer", organization="acme")
    with WorkspaceContext("team-a"):
        assert store.get_skill_version("reviewer", 1, organization="acme").version == 1
        assert store.delete_skill_and_collect_artifacts("reviewer", "acme") == [team_a_path]
