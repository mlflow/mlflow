from functools import partial

import pytest

from mlflow.entities import SkillSourceType, SkillStatus
from mlflow.exceptions import MlflowException
from mlflow.utils.workspace_context import WorkspaceContext

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture
def workspace_store(store, workspaces_enabled):
    if not workspaces_enabled:
        pytest.skip("Workspace isolation is only applicable when workspaces are enabled")
    return store


def _create_skill(store, team):
    store.create_skill("reviewer", organization="acme", description=team, created_by=team)
    for _ in range(2):
        store.create_skill_version(
            "reviewer",
            organization="acme",
            status=SkillStatus.DRAFT,
            source_type=SkillSourceType.GIT,
            source="https://example.com/reviewer.git",
            created_by=team,
        )
    store.set_skill_tag("reviewer", "team", team, organization="acme")
    store.set_skill_version_tag("reviewer", 1, "team", team, organization="acme")
    store.set_skill_alias("reviewer", "preview", 1, organization="acme")
    return _snapshot(store)


def _snapshot(store):
    return (
        store.get_skill("reviewer", organization="acme"),
        list(
            store.search_skill_versions("reviewer", organization="acme", order_by=["version ASC"])
        ),
    )


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("get_skill", {}),
        ("get_skill_version", {"version": 1}),
        ("get_latest_skill_version", {}),
        ("get_skill_version_by_alias", {"alias": "preview"}),
        ("update_skill", {"description": "changed"}),
        ("update_skill_version", {"version": 1, "status": SkillStatus.ACTIVE}),
        ("delete_skill_version", {"version": 1}),
        ("set_skill_tag", {"key": "team", "value": "changed"}),
        ("delete_skill_tag", {"key": "team"}),
        ("set_skill_version_tag", {"version": 1, "key": "team", "value": "changed"}),
        ("delete_skill_version_tag", {"version": 1, "key": "team"}),
        ("set_skill_alias", {"alias": "preview", "version": 2}),
        ("delete_skill_alias", {"alias": "preview"}),
        ("delete_skill", {}),
        ("delete_skill_and_collect_artifacts", {}),
    ],
)
def test_skill_operations_cannot_access_another_workspace(workspace_store, method, kwargs):
    store = workspace_store
    with WorkspaceContext("team-a"):
        original = _create_skill(store, "team-a")

    with WorkspaceContext("team-b"):
        with pytest.raises(MlflowException, match="not found") as exc:
            getattr(store, method)("reviewer", organization="acme", **kwargs)
        assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
        assert store.search_skills() == []
        assert store.search_skill_versions("reviewer", organization="acme") == []

    with WorkspaceContext("team-a"):
        assert _snapshot(store) == original


@pytest.mark.parametrize(
    ("method", "kwargs", "entity", "field", "expected"),
    [
        ("update_skill", {"description": "changed"}, "parent", "description", "changed"),
        (
            "update_skill_version",
            {"version": 1, "status": SkillStatus.ACTIVE},
            "version",
            "status",
            SkillStatus.ACTIVE,
        ),
        (
            "set_skill_tag",
            {"key": "team", "value": "changed"},
            "parent",
            "tags",
            {"team": "changed"},
        ),
        ("delete_skill_tag", {"key": "team"}, "parent", "tags", {}),
        (
            "set_skill_version_tag",
            {"version": 1, "key": "team", "value": "changed"},
            "version",
            "tags",
            {"team": "changed"},
        ),
        ("delete_skill_version_tag", {"version": 1, "key": "team"}, "version", "tags", {}),
        (
            "set_skill_alias",
            {"alias": "preview", "version": 2},
            "parent",
            "aliases",
            {"preview": 2},
        ),
        ("delete_skill_alias", {"alias": "preview"}, "parent", "aliases", {}),
    ],
)
def test_skill_mutations_only_change_the_current_workspace(
    workspace_store, method, kwargs, entity, field, expected
):
    store = workspace_store
    with WorkspaceContext("team-a"):
        original = _create_skill(store, "team-a")
    with WorkspaceContext("team-b"):
        _create_skill(store, "team-b")
        getattr(store, method)("reviewer", organization="acme", **kwargs)
        parent, versions = _snapshot(store)
        result = parent if entity == "parent" else versions[0]
        assert result.workspace == "team-b"
        assert getattr(result, field) == expected

    with WorkspaceContext("team-a"):
        assert _snapshot(store) == original
        assert (
            store.get_skill_version_by_alias("reviewer", "preview", organization="acme")
            == original[1][0]
        )


@pytest.mark.parametrize(
    "method", ["delete_skill_version", "delete_skill", "delete_skill_and_collect_artifacts"]
)
def test_skill_deletion_preserves_same_identity_in_another_workspace(workspace_store, method):
    store = workspace_store
    with WorkspaceContext("team-a"):
        original = _create_skill(store, "team-a")
    with WorkspaceContext("team-b"):
        _create_skill(store, "team-b")
        kwargs = {"version": 1} if method == "delete_skill_version" else {}
        result = getattr(store, method)("reviewer", organization="acme", **kwargs)
        assert result == ([] if method == "delete_skill_and_collect_artifacts" else None)
        with pytest.raises(MlflowException, match="not found") as exc:
            store.get_skill_version("reviewer", 1, organization="acme")
        assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
        if method == "delete_skill_version":
            parent, versions = _snapshot(store)
            assert parent.aliases == {}
            assert parent.latest_version == 2
            assert [version.version for version in versions] == [2]
        else:
            assert store.search_skills() == []
            assert store.search_skill_versions("reviewer", organization="acme") == []

    with WorkspaceContext("team-a"):
        assert _snapshot(store) == original
        assert (
            store.get_skill_version_by_alias("reviewer", "preview", organization="acme")
            == original[1][0]
        )


@pytest.mark.parametrize("target", ["parent", "version"])
def test_skill_search_filters_and_pages_are_workspace_scoped(workspace_store, target):
    store = workspace_store
    for team in ["team-a", "team-b"]:
        with WorkspaceContext(team):
            _create_skill(store, team)
            store.create_skill_version("writer", organization="acme")

    with WorkspaceContext("team-b"):
        search = (
            store.search_skills
            if target == "parent"
            else partial(store.search_skill_versions, "reviewer", organization="acme")
        )
        assert search(filter_string="tags.team = 'team-a'") == []
        matches = search(filter_string="tags.team = 'team-b'")
        assert len(matches) == 1
        assert matches[0].workspace == "team-b"
        first = search(max_results=1)
        assert len(first) == 1
        assert first.token is not None
        second = search(max_results=1, page_token=first.token)
        assert len(second) == 1
        assert second.token is None
        assert [item.workspace for item in [*first, *second]] == ["team-b", "team-b"]
        assert [
            item.name if target == "parent" else item.version for item in [*first, *second]
        ] == (["reviewer", "writer"] if target == "parent" else [1, 2])
