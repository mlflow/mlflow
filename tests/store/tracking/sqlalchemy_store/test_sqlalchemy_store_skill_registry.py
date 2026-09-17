import pytest

from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_source import (
    GitSource,
    MlflowSource,
    OCISource,
    SkillSourceType,
    ZipSource,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.utils.workspace_context import WorkspaceContext

pytestmark = pytest.mark.notrackingurimock


def test_create_and_get_skill(store):
    created = store.create_skill(
        "reviewer",
        organization="acme",
        description="Reviews code",
        created_by="alice",
    )

    assert created.name == "reviewer"
    assert created.organization == "acme"
    assert created.description == "Reviews code"
    assert created.status is None
    assert created.created_by == "alice"
    assert created.last_updated_by == "alice"
    assert created.creation_timestamp is not None

    retrieved = store.get_skill("reviewer", organization="acme")
    assert retrieved == created


def test_create_skill_duplicate_raises(store):
    store.create_skill("reviewer", organization="acme")

    with pytest.raises(MlflowException, match="already exists") as exc:
        store.create_skill("reviewer", organization="acme")

    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_same_skill_name_is_allowed_in_different_organizations(store):
    acme = store.create_skill("reviewer", organization="acme")
    example = store.create_skill("reviewer", organization="example")

    assert acme.organization == "acme"
    assert example.organization == "example"


def test_get_skill_not_found_raises(store):
    with pytest.raises(MlflowException, match="not found") as exc:
        store.get_skill("reviewer", organization="acme")

    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_update_skill_distinguishes_omitted_and_null_values(store):
    store.create_skill("reviewer", description="Reviews code", created_by="alice")

    unchanged = store.update_skill("reviewer")
    assert unchanged.description == "Reviews code"

    cleared = store.update_skill("reviewer", description=None, last_updated_by="bob")
    assert cleared.description is None
    assert cleared.last_updated_by == "bob"


def test_skill_icons_round_trip_and_can_be_cleared(store):
    icons = [{"src": "https://example.com/reviewer.svg", "sizes": ["any"]}]
    created = store.create_skill("reviewer", icons=icons)
    assert created.icons == icons

    updated = store.update_skill("reviewer", icons=None)
    assert updated.icons is None

    unchanged = store.update_skill("reviewer", description="Updated")
    assert unchanged.icons is None


def test_search_skills_returns_stable_paginated_results(store):
    store.create_skill("writer", organization="acme")
    store.create_skill("reviewer", organization="acme")

    first_page = store.search_skills(max_results=1)
    assert [skill.name for skill in first_page] == ["reviewer"]
    assert first_page.token is not None

    second_page = store.search_skills(max_results=1, page_token=first_page.token)
    assert [skill.name for skill in second_page] == ["writer"]
    assert second_page.token is None


@pytest.mark.parametrize("name", ["", "Reviewer", "reviewer_name", "reviewer--tool"])
def test_create_skill_rejects_invalid_name(store, name):
    with pytest.raises(MlflowException, match="Invalid skill name|must not be empty"):
        store.create_skill(name)


def test_skill_identity_is_workspace_scoped(store, workspaces_enabled):
    if not workspaces_enabled:
        pytest.skip("Workspace isolation is only applicable when workspaces are enabled")

    with WorkspaceContext("team-a"):
        store.create_skill("reviewer", organization="acme")

    with WorkspaceContext("team-b"):
        store.create_skill("reviewer", organization="acme")
        assert store.get_skill("reviewer", organization="acme").workspace == "team-b"

    with WorkspaceContext("team-a"):
        assert store.get_skill("reviewer", organization="acme").workspace == "team-a"


def _persist_skill_version(store, version=1, **kwargs):
    with store.ManagedSessionMaker(read_only=False) as session:
        return store._persist_skill_version(
            session=session,
            name="reviewer",
            organization="acme",
            version=version,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("source_type", "source", "expected_source"),
    [
        (
            SkillSourceType.GIT,
            {
                "source": "https://github.com/acme/skills.git",
                "ref": "v1.0.0",
                "subpath": "reviewer",
            },
            GitSource(url="https://github.com/acme/skills.git", ref="v1.0.0", subpath="reviewer"),
        ),
        (
            SkillSourceType.OCI,
            {"source": "registry.example.com/skills/reviewer", "subpath": "skill"},
            OCISource(image="registry.example.com/skills/reviewer", subpath="skill"),
        ),
        (
            SkillSourceType.ZIP,
            {"source": "https://example.com/reviewer.zip", "subpath": "reviewer"},
            ZipSource(url="https://example.com/reviewer.zip", subpath="reviewer"),
        ),
        (
            SkillSourceType.MLFLOW,
            {"source": "mlflow-artifacts:/skills/reviewer", "subpath": "reviewer"},
            MlflowSource(artifact_path="mlflow-artifacts:/skills/reviewer", subpath="reviewer"),
        ),
    ],
)
def test_skill_version_source_round_trip(store, source_type, source, expected_source):
    created = _persist_skill_version(
        store,
        source_type=source_type,
        digest="sha256:abc",
        **source,
    )

    assert created.source_type == source_type
    assert created.source == expected_source
    assert created.digest == "sha256:abc"

    retrieved = store.get_skill_version("reviewer", 1, organization="acme")
    assert retrieved.source == expected_source
    assert retrieved.digest == "sha256:abc"


def test_skill_version_auto_creates_parent_and_preserves_existing_parent(store):
    created = _persist_skill_version(store, status=SkillStatus.DRAFT.value)
    assert created.status == SkillStatus.DRAFT

    parent = store.get_skill("reviewer", organization="acme")
    assert parent.description is None

    store.update_skill(
        "reviewer",
        organization="acme",
        description="Reviews code",
        icons=[{"src": "https://example.com/reviewer.svg"}],
    )
    _persist_skill_version(store, version=2)
    parent = store.get_skill("reviewer", organization="acme")
    assert parent.description == "Reviews code"
    assert parent.icons == [{"src": "https://example.com/reviewer.svg"}]


def test_skill_version_identity_is_workspace_scoped(store, workspaces_enabled):
    if not workspaces_enabled:
        pytest.skip("Workspace isolation is only applicable when workspaces are enabled")

    with WorkspaceContext("team-a"):
        version_a = store.create_skill_version("reviewer", organization="acme")

    with WorkspaceContext("team-b"):
        version_b = store.create_skill_version("reviewer", organization="acme")

    assert version_a.version == 1
    assert version_b.version == 1

    with WorkspaceContext("team-a"):
        assert store.get_skill_version("reviewer", 1, organization="acme").workspace == "team-a"

    with WorkspaceContext("team-b"):
        assert store.get_skill_version("reviewer", 1, organization="acme").workspace == "team-b"


def test_duplicate_skill_version_raises(store):
    _persist_skill_version(store)

    with pytest.raises(MlflowException, match="already exists") as exc:
        _persist_skill_version(store)

    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_create_skill_version_allocates_monotonically(store):
    first = store.create_skill_version("reviewer", organization="acme")
    second = store.create_skill_version("reviewer", organization="acme")

    assert first.version == 1
    assert second.version == 2


def test_create_skill_version_does_not_reuse_deleted_version(store):
    _persist_skill_version(store, status=SkillStatus.DELETED.value)

    created = store.create_skill_version("reviewer", organization="acme")

    assert created.version == 2


def test_create_skill_version_retries_and_rolls_back_after_conflict(store, monkeypatch):
    original_persist = store._persist_skill_version
    persist_calls = 0

    def persist_with_conflict_after_insert(*args, **kwargs):
        nonlocal persist_calls
        persist_calls += 1
        created = original_persist(*args, **kwargs)
        if persist_calls == 1:
            raise MlflowException(
                "simulated version conflict",
                error_code=RESOURCE_ALREADY_EXISTS,
            )
        return created

    monkeypatch.setattr(store, "_persist_skill_version", persist_with_conflict_after_insert)

    created = store.create_skill_version("reviewer", organization="acme")

    assert created.version == 1
    assert persist_calls == 2


def test_deleted_skill_version_is_not_retrievable(store):
    _persist_skill_version(store, status=SkillStatus.DELETED.value)

    with pytest.raises(MlflowException, match="not found") as exc:
        store.get_skill_version("reviewer", 1, organization="acme")

    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
