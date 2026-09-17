import pytest

from mlflow.exceptions import MlflowException
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
