from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.server import SERVE_ARTIFACTS_ENV_VAR, handlers
from mlflow.server.skill_registry import (
    SkillVersionRegistration,
    delete_skill,
    register_skill_version,
)
from mlflow.server.skill_registry import deletion as deletion_module
from mlflow.store.tracking.dbmodels.models import (
    SqlAgentPlugin,
    SqlAgentPluginVersion,
    SqlAgentPluginVersionMember,
)
from mlflow.store.tracking.skill_registry.artifact_paths import validate_referenced_mlflow_source

from tests.server.skill_registry.conftest import (
    SKILL_FILES,
    skill_archive,
    stored_files,
    version_rows,
)

pytestmark = pytest.mark.notrackingurimock

_PACKAGE_PATH = "agent-plugins/@acme/toolkit/0123456789abcdef0123456789abcdef"


def _upload(name="reviewer", organization="acme"):
    version = register_skill_version(
        SkillVersionRegistration(name=name, organization=organization),
        content=skill_archive(),
        multipart=True,
    )
    return version, version.source.artifact_path.removeprefix("mlflow-artifacts:/")


def _import_member(store, artifact_root, name="reviewer"):
    """Persist a member that references a plugin's stored tree, as packaged import does."""
    tree = artifact_root / _PACKAGE_PATH
    (tree / "skills" / name).mkdir(parents=True, exist_ok=True)
    (tree / "skills" / name / "SKILL.md").write_bytes(SKILL_FILES["SKILL.md"])
    (tree / "plugin.json").write_bytes(b"{}")
    source, subpath = validate_referenced_mlflow_source(
        f"mlflow-artifacts:/{_PACKAGE_PATH}", f"skills/{name}"
    )
    return store.create_skill_version(
        name, organization="acme", source_type="mlflow", source=source, subpath=subpath
    )


def test_delete_skill_reclaims_every_owned_path(store, artifact_root):
    _, first = _upload()
    _, second = _upload()
    assert stored_files(artifact_root / first) == sorted(SKILL_FILES)

    delete_skill("reviewer", organization="acme")

    assert not (artifact_root / first).exists()
    assert not (artifact_root / second).exists()
    assert version_rows(store) == 0
    with pytest.raises(MlflowException, match="not found"):
        store.get_skill("reviewer", organization="acme")


def test_delete_skill_commits_the_database_before_touching_artifacts(store, artifact_root):
    _, path = _upload()
    observed = []

    def check_then_delete(artifact_path):
        # By the time any content is removed, the rows are already gone for other sessions.
        observed.append((artifact_path, version_rows(store), (artifact_root / path).exists()))
        return True

    with mock.patch.object(
        deletion_module, "delete_artifact_tree_best_effort", side_effect=check_then_delete
    ) as cleanup:
        delete_skill("reviewer", organization="acme")
    cleanup.assert_called_once_with(path)
    assert observed == [(path, 0, True)]


def test_cleanup_failure_does_not_restore_rows_or_fail_the_delete(store, artifact_root):
    _, first = _upload()
    _, second = _upload()
    repo_delete_calls = []

    def fail_first(artifact_path):
        repo_delete_calls.append(artifact_path)
        if len(repo_delete_calls) == 1:
            raise OSError("storage unavailable")

    repo = handlers._get_artifact_repo_mlflow_artifacts()
    real_delete = repo.delete_artifacts
    with mock.patch.object(
        repo,
        "delete_artifacts",
        side_effect=lambda p: fail_first(p) or real_delete(p),
    ) as repo_delete:
        delete_skill("reviewer", organization="acme")
    assert repo_delete.call_count == 2
    # The delete stands, the failed path is an accepted leak, and the other path is reclaimed.
    assert version_rows(store) == 0
    leaked = [p for p in (first, second) if (artifact_root / p).exists()]
    assert leaked == [repo_delete_calls[0]]


def test_delete_skill_never_deletes_a_referenced_package_tree(store, artifact_root):
    # The member is the last reference to the plugin's tree; deleting the skill still leaves
    # it alone, because the tree belongs to the plugin version, not to the skill.
    _import_member(store, artifact_root)
    _, owned = _upload()
    package_files = stored_files(artifact_root / _PACKAGE_PATH)

    with mock.patch.object(
        deletion_module,
        "delete_artifact_tree_best_effort",
        wraps=deletion_module.delete_artifact_tree_best_effort,
    ) as cleanup:
        delete_skill("reviewer", organization="acme")
    cleanup.assert_called_once_with(owned)
    assert version_rows(store) == 0
    assert stored_files(artifact_root / _PACKAGE_PATH) == package_files
    assert not (artifact_root / owned).exists()


def test_delete_skill_with_only_references_touches_no_artifacts(store, artifact_root):
    _import_member(store, artifact_root)
    before = stored_files(artifact_root)
    with mock.patch.object(deletion_module, "delete_artifact_tree_best_effort") as cleanup:
        delete_skill("reviewer", organization="acme")
    cleanup.assert_not_called()
    assert stored_files(artifact_root) == before


def test_delete_skill_leaves_unrelated_content_alone(store, artifact_root):
    _, deleted_path = _upload(name="reviewer")
    _, sibling_name = _upload(name="reviewer-two")
    _, sibling_org = _upload(name="reviewer", organization="example")
    unrelated = artifact_root / "0" / "run" / "artifacts" / "model.pkl"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"model")

    delete_skill("reviewer", organization="acme")

    assert not (artifact_root / deleted_path).exists()
    assert stored_files(artifact_root / sibling_name) == sorted(SKILL_FILES)
    assert stored_files(artifact_root / sibling_org) == sorted(SKILL_FILES)
    assert unrelated.read_bytes() == b"model"
    # Only the token directory goes, never the identity prefix of a surviving neighbor.
    assert (artifact_root / "skills" / "@acme" / "reviewer-two").is_dir()


def test_delete_then_recreate_restarts_versions_at_a_fresh_path(store, artifact_root):
    for _ in range(3):
        _upload()
    _, last_old_path = _upload()
    delete_skill("reviewer", organization="acme")

    recreated, new_path = _upload()

    assert recreated.version == 1
    assert new_path != last_old_path
    assert stored_files(artifact_root / new_path) == sorted(SKILL_FILES)


def test_blocked_delete_removes_neither_rows_nor_artifacts(store, artifact_root):
    _, path = _upload()
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(store._with_workspace_field(SqlAgentPlugin(organization="acme", name="kit")))
        session.add(
            store._with_workspace_field(
                SqlAgentPluginVersion(
                    organization="acme",
                    name="kit",
                    version="1.0.0",
                    version_major=1,
                    version_minor=0,
                    version_patch=0,
                    version_prerelease_sort_key="",
                    plugin_json={"name": "kit", "version": "1.0.0"},
                )
            )
        )
        session.flush()
        session.add(
            SqlAgentPluginVersionMember(
                plugin_workspace="default",
                plugin_organization="acme",
                plugin_name="kit",
                plugin_version="1.0.0",
                member_name="reviewer",
                member_organization="acme",
                member_version=1,
            )
        )

    with mock.patch.object(deletion_module, "delete_artifact_tree_best_effort") as cleanup:
        with pytest.raises(MlflowException, match="live agent plugin versions") as exc:
            delete_skill("reviewer", organization="acme")
    cleanup.assert_not_called()
    assert exc.value.error_code == "RESOURCE_CONFLICT"
    assert version_rows(store) == 1
    assert stored_files(artifact_root / path) == sorted(SKILL_FILES)


def test_delete_skill_without_artifact_serving_still_deletes_rows(
    store, artifact_root, monkeypatch
):
    # Content was uploaded while artifacts were served; the deployment later stopped serving
    # them. The registry delete must still work, leaving the bytes for the operator.
    _, path = _upload()
    monkeypatch.setenv(SERVE_ARTIFACTS_ENV_VAR, "false")

    with mock.patch.object(deletion_module, "delete_artifact_tree_best_effort") as cleanup:
        delete_skill("reviewer", organization="acme")
    cleanup.assert_not_called()
    assert version_rows(store) == 0
    assert stored_files(artifact_root / path) == sorted(SKILL_FILES)


def test_delete_remote_skill_without_artifact_serving(store, no_artifact_serving):
    register_skill_version(
        SkillVersionRegistration(name="reviewer", source="https://example.com/r.git")
    )
    delete_skill("reviewer")
    assert version_rows(store) == 0
