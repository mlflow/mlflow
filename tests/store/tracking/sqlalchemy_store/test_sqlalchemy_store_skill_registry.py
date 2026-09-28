from types import SimpleNamespace

import pytest
import sqlalchemy
from sqlalchemy.dialects import mssql
from sqlalchemy.orm import Session

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
from mlflow.store.tracking.dbmodels.models import (
    SqlSkill,
    SqlSkillAlias,
    SqlSkillTag,
    SqlSkillVersion,
)
from mlflow.store.tracking.skill_registry.sqlalchemy_mixin import SqlAlchemySkillRegistryMixin
from mlflow.store.tracking.skill_registry_pagination import SkillRegistryPaginationToken
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


def test_update_skill_preserves_resolved_fields(store):
    store.create_skill_version("reviewer")
    store.create_skill_version("reviewer", status=SkillStatus.DRAFT.value)

    updated = store.update_skill("reviewer", description="Updated")
    assert updated.latest_version == 1
    assert updated.status == SkillStatus.ACTIVE

    updated = store.update_skill("reviewer", icons=[{"src": "https://example.com/reviewer.svg"}])
    assert updated.latest_version == 1
    assert updated.status == SkillStatus.ACTIVE


def test_skill_icons_round_trip_and_can_be_cleared(store):
    icons = [{"src": "https://example.com/reviewer.svg", "sizes": ["any"]}]
    created = store.create_skill("reviewer", icons=icons)
    assert created.icons == icons

    updated = store.update_skill("reviewer", icons=None)
    assert updated.icons is None

    unchanged = store.update_skill("reviewer", description="Updated")
    assert unchanged.icons is None


def _get_skill_search_text(store, name="reviewer", organization=""):
    with store.ManagedSessionMaker() as session:
        return (
            store
            ._get_query(session, SqlSkill)
            .filter(SqlSkill.name == name, SqlSkill.organization == organization)
            .one()
            .search_text
        )


def test_skill_search_text_is_persisted_and_recomputed_for_description_only(store):
    store.create_skill(
        "reviewer",
        organization="acme",
        description="Reviews code",
        icons=[{"src": "https://example.com/reviewer.svg"}],
    )
    assert _get_skill_search_text(store, organization="acme") == "reviewer Reviews code"

    store.update_skill(
        "reviewer",
        organization="acme",
        icons=[{"src": "https://example.com/reviewer-dark.svg"}],
    )
    assert _get_skill_search_text(store, organization="acme") == "reviewer Reviews code"

    store.update_skill("reviewer", organization="acme", description="Audits pull requests")
    assert _get_skill_search_text(store, organization="acme") == "reviewer Audits pull requests"


def test_auto_created_skill_version_parent_gets_search_text(store):
    store.create_skill_version("reviewer", organization="acme")

    assert _get_skill_search_text(store, organization="acme") == "reviewer"


def test_search_skills_returns_stable_paginated_results(store):
    store.create_skill("writer", organization="acme")
    store.create_skill("reviewer", organization="acme")

    first_page = store.search_skills(max_results=1)
    assert [skill.name for skill in first_page] == ["reviewer"]
    assert first_page.token is not None

    second_page = store.search_skills(max_results=1, page_token=first_page.token)
    assert [skill.name for skill in second_page] == ["writer"]
    assert second_page.token is None


def test_search_skills_token_is_bound_to_query(store):
    for name in ["auditor", "reviewer", "writer"]:
        store.create_skill(name, organization="acme")

    first_page = store.search_skills(
        filter_string="organization = 'acme'",
        order_by=["name ASC"],
        max_results=1,
    )
    assert [skill.name for skill in first_page] == ["auditor"]
    assert first_page.token is not None

    decoded = SkillRegistryPaginationToken.decode(first_page.token)
    assert decoded.query_scope == "skills"
    assert decoded.offset == 1

    second_page = store.search_skills(
        filter_string="organization = 'acme'",
        order_by=["name ASC"],
        max_results=2,
        page_token=first_page.token,
    )
    assert [skill.name for skill in second_page] == ["reviewer", "writer"]

    with pytest.raises(MlflowException, match="different order_by"):
        store.search_skills(
            filter_string="organization = 'acme'",
            order_by=["name DESC"],
            page_token=first_page.token,
        )


def test_search_skills_filters_by_derived_status_organization_tags_and_search_text(store):
    store.create_skill("reviewer", organization="acme", description="Reviews pull requests")
    store.create_skill_version("reviewer", organization="acme")
    store.create_skill_version("reviewer", organization="acme", status=SkillStatus.DRAFT.value)
    store.set_skill_tag("reviewer", "team", "platform", organization="acme")
    store.set_skill_tag("reviewer", "priority", "high", organization="acme")

    store.create_skill("draft-helper", organization="acme", description="Draft assistance")
    store.create_skill_version(
        "draft-helper", organization="acme", status=SkillStatus.DRAFT.value
    )
    store.set_skill_tag("draft-helper", "team", "platform", organization="acme")

    store.create_skill("reviewer", organization="beta", description="Reviews beta code")
    store.create_skill_version("reviewer", organization="beta")
    store.set_skill_tag("reviewer", "team", "platform", organization="beta")

    active_acme_platform = store.search_skills(
        filter_string=(
            "organization = 'acme' AND status = 'active' AND tags.team = 'platform'"
        )
    )
    assert [(skill.organization, skill.name) for skill in active_acme_platform] == [
        ("acme", "reviewer")
    ]

    draft_acme = store.search_skills(
        filter_string="organization = 'acme' AND status = 'draft'",
    )
    assert [skill.name for skill in draft_acme] == ["draft-helper"]

    text_matches = store.search_skills(
        filter_string="search_text ILIKE '%pull%'",
        order_by=["name DESC"],
    )
    assert [(skill.organization, skill.name) for skill in text_matches] == [
        ("acme", "reviewer")
    ]


def test_search_skills_filters_source_type_by_latest_resolved_version(store):
    store.create_skill_version(
        "git-reviewer",
        organization="acme",
        source_type=SkillSourceType.GIT,
        source="https://example.com/skills.git",
    )
    store.create_skill_version(
        "git-reviewer",
        organization="acme",
        source_type=SkillSourceType.ZIP,
        source="https://example.com/skill.zip",
        status=SkillStatus.DRAFT.value,
    )
    store.create_skill_version(
        "draft-zip-reviewer",
        organization="acme",
        source_type=SkillSourceType.GIT,
        source="https://example.com/draft.git",
        status=SkillStatus.DRAFT.value,
    )
    store.create_skill_version(
        "draft-zip-reviewer",
        organization="acme",
        source_type=SkillSourceType.ZIP,
        source="https://example.com/draft.zip",
        status=SkillStatus.DRAFT.value,
    )
    store.create_skill_version(
        "deleted-git-reviewer",
        organization="acme",
        source_type=SkillSourceType.GIT,
        source="https://example.com/deleted.git",
        status=SkillStatus.DRAFT.value,
    )
    store.delete_skill_version("deleted-git-reviewer", 1, organization="acme")

    git_matches = store.search_skills(
        filter_string="organization = 'acme' AND source_type = 'git'",
        order_by=["name ASC"],
    )
    assert [skill.name for skill in git_matches] == ["git-reviewer"]

    zip_matches = store.search_skills(
        filter_string="organization = 'acme' AND source_type = 'zip'",
        order_by=["name ASC"],
    )
    assert [skill.name for skill in zip_matches] == ["draft-zip-reviewer"]

    zip_ordered_matches = store.search_skills(
        filter_string="organization = 'acme' AND source_type != 'git'",
        order_by=["source_type ASC", "name ASC"],
    )
    assert [skill.name for skill in zip_ordered_matches] == ["draft-zip-reviewer"]


@pytest.mark.parametrize(
    "max_results",
    [0, -1, 1001, True, "1"],
)
def test_search_skills_rejects_invalid_max_results(store, max_results):
    with pytest.raises(MlflowException, match="max_results") as exc:
        store.search_skills(max_results=max_results)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_search_skills_eager_loads_tags_and_aliases(store):
    for index in range(20):
        store.create_skill(f"skill-{index:02d}")

    statements = []

    def capture_statement(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    sqlalchemy.event.listen(store.engine, "before_cursor_execute", capture_statement)
    try:
        skills = store.search_skills()
    finally:
        sqlalchemy.event.remove(store.engine, "before_cursor_execute", capture_statement)

    assert len(skills) == 20
    assert len(statements) == 3


def test_skill_query_relationship_loading_is_sql_server_compatible():
    engine = sqlalchemy.create_engine("sqlite:///:memory:")
    try:
        SqlSkill.metadata.create_all(
            engine,
            tables=[
                model.__table__ for model in (SqlSkill, SqlSkillVersion, SqlSkillTag, SqlSkillAlias)
            ],
        )
        with engine.begin() as connection:
            connection.execute(
                SqlSkill.__table__.insert().values(
                    workspace="default", organization="acme", name="reviewer"
                )
            )

        owner = SimpleNamespace(_get_query=lambda session, model: session.query(model))
        captured = []
        with Session(engine) as session:

            @sqlalchemy.event.listens_for(session, "do_orm_execute")
            def capture_tag_query(state):
                if state.is_relationship_load and state.bind_mapper.class_ is SqlSkillTag:
                    statement = state.statement.params(**(state.parameters or {}))
                    sql = str(
                        statement.compile(
                            dialect=mssql.dialect(), compile_kwargs={"literal_binds": True}
                        )
                    )
                    captured.append(" ".join(sql.split()))

            query = SqlAlchemySkillRegistryMixin._skill_query(owner, session)
            query.filter(SqlSkill.name == "reviewer", SqlSkill.organization == "acme").one()

        assert len(captured) == 1
        unsupported = "WHERE (skill_tags.workspace, skill_tags.organization, skill_tags.name) IN ("
        assert unsupported not in captured[0], captured[0]
    finally:
        engine.dispose()


def test_skill_queries_load_relationships_on_supported_backends(store):
    store.create_skill("reviewer", organization="acme")

    retrieved = store.get_skill("reviewer", organization="acme")
    assert retrieved.name == "reviewer"

    updated = store.update_skill("reviewer", organization="acme", description="Updated")
    assert updated.description == "Updated"

    listed = store.search_skills()
    assert [(skill.name, skill.organization) for skill in listed] == [("reviewer", "acme")]


def test_skill_tags_can_be_set_updated_deleted_and_filtered(store):
    store.create_skill("reviewer", organization="acme")
    store.create_skill_version("reviewer", organization="acme")

    store.set_skill_tag("reviewer", "team", "platform", organization="acme")
    assert store.get_skill("reviewer", organization="acme").tags == {"team": "platform"}

    store.set_skill_tag("reviewer", "team", "ml-platform", organization="acme")
    assert store.get_skill("reviewer", organization="acme").tags == {"team": "ml-platform"}
    assert [skill.name for skill in store.search_skills("tags.team LIKE 'ml-%'")] == [
        "reviewer"
    ]

    store.delete_skill_tag("reviewer", "team", organization="acme")
    assert store.get_skill("reviewer", organization="acme").tags == {}

    with pytest.raises(MlflowException, match="Tag 'team' not found"):
        store.delete_skill_tag("reviewer", "team", organization="acme")


def test_set_skill_tag_validates_parent_and_tag_payload(store):
    with pytest.raises(MlflowException, match="not found"):
        store.set_skill_tag("reviewer", "team", "platform", organization="acme")

    with pytest.raises(MlflowException, match="not found"):
        store.delete_skill_tag("reviewer", "team", organization="acme")

    store.create_skill("reviewer", organization="acme")
    with pytest.raises(MlflowException, match="key"):
        store.set_skill_tag("reviewer", None, "platform", organization="acme")


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


def _persist_deleted_skill_version(store):
    _persist_skill_version(store)
    with store.ManagedSessionMaker(read_only=False) as session:
        skill_version = (
            store
            ._get_query(session, SqlSkillVersion)
            .filter(
                SqlSkillVersion.name == "reviewer",
                SqlSkillVersion.organization == "acme",
                SqlSkillVersion.version == 1,
            )
            .one()
        )
        skill_version.status = SkillStatus.DELETED.value


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
    digest = "a" * 64
    created = _persist_skill_version(
        store,
        source_type=source_type,
        digest=digest,
        **source,
    )

    assert created.source_type == source_type
    assert created.source == expected_source
    assert created.digest == digest

    retrieved = store.get_skill_version("reviewer", 1, organization="acme")
    assert retrieved.source == expected_source
    assert retrieved.digest == digest


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


def test_create_skill_version_persists_created_by_for_parent_and_version(store):
    created = store.create_skill_version("reviewer", created_by="alice")

    assert created.created_by == "alice"
    assert created.last_updated_by == "alice"

    parent = store.get_skill("reviewer")
    assert parent.created_by == "alice"
    assert parent.last_updated_by == "alice"


def test_create_skill_version_does_not_update_existing_parent_audit_fields(store):
    store.create_skill("reviewer", created_by="alice")

    created = store.create_skill_version("reviewer", created_by="bob")

    assert created.created_by == "bob"
    assert created.last_updated_by == "bob"
    parent = store.get_skill("reviewer")
    assert parent.created_by == "alice"
    assert parent.last_updated_by == "alice"


def test_create_skill_version_does_not_reuse_deleted_version(store):
    _persist_deleted_skill_version(store)

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


def test_create_skill_version_failure_does_not_leave_orphan_parent(store):
    with pytest.raises(MlflowException, match="Invalid Skill source type") as exc:
        store.create_skill_version(
            "orphan-skill",
            source_type="svn",
            source="https://example.com/skill",
        )

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"

    with pytest.raises(MlflowException, match="not found"):
        store.get_skill("orphan-skill")


@pytest.mark.parametrize("status", [SkillStatus.DELETED.value, SkillStatus.DEPRECATED.value])
def test_create_skill_version_rejects_non_registration_status(store, status):
    with pytest.raises(MlflowException, match="must have status 'active' or 'draft'") as exc:
        store.create_skill_version("invalid-status", status=status)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    with pytest.raises(MlflowException, match="not found"):
        store.get_skill("invalid-status")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"source_type": "git", "source": "x" * 2049},
            "source must be",
        ),
        (
            {"source_type": "git", "source": "https://example.com/skill", "ref": "x" * 2049},
            "ref must be",
        ),
        (
            {
                "source_type": "git",
                "source": "https://example.com/skill",
                "subpath": "x" * 2049,
            },
            "subpath must be",
        ),
        (
            {"source_type": "git", "source": "https://example.com/skill", "digest": "a" * 200},
            "digest must be",
        ),
        (
            {
                "source_type": SkillSourceType.ZIP,
                "source": "https://example.com/skill.zip",
                "ref": "v1",
            },
            "ref is only supported",
        ),
    ],
)
def test_create_skill_version_rejects_invalid_source_metadata(store, kwargs, message):
    with pytest.raises(MlflowException, match=message) as exc:
        store.create_skill_version("invalid-skill", **kwargs)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    with pytest.raises(MlflowException, match="not found"):
        store.get_skill("invalid-skill")


def test_deleted_skill_version_is_not_retrievable(store):
    _persist_deleted_skill_version(store)

    with pytest.raises(MlflowException, match="not found") as exc:
        store.get_skill_version("reviewer", 1, organization="acme")

    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_search_skill_versions_filters_by_stored_status_source_digest_and_tags(store):
    active_digest = "a" * 64
    draft_digest = "b" * 64
    deleted_digest = "c" * 64
    store.create_skill_version(
        "reviewer",
        organization="acme",
        source_type=SkillSourceType.GIT,
        source="https://example.com/skills.git",
        digest=active_digest,
    )
    store.create_skill_version(
        "reviewer",
        organization="acme",
        source_type=SkillSourceType.ZIP,
        source="https://example.com/skill.zip",
        digest=draft_digest,
        status=SkillStatus.DRAFT.value,
    )
    store.create_skill_version(
        "reviewer",
        organization="acme",
        source_type=SkillSourceType.GIT,
        source="https://example.com/skills.git",
        digest=deleted_digest,
        status=SkillStatus.DRAFT.value,
    )
    store.set_skill_version_tag("reviewer", 2, "release", "canary", organization="acme")
    store.delete_skill_version("reviewer", 3, organization="acme")

    assert [
        version.version
        for version in store.search_skill_versions(
            "reviewer",
            organization="acme",
            filter_string="source_type = 'git'",
            order_by=["version DESC"],
        )
    ] == [1]
    assert [
        version.version
        for version in store.search_skill_versions(
            "reviewer",
            organization="acme",
            filter_string="status = 'draft'",
        )
    ] == [2]
    assert [
        version.version
        for version in store.search_skill_versions(
            "reviewer",
            organization="acme",
            filter_string=f"digest = '{draft_digest}'",
        )
    ] == [2]
    assert [
        version.version
        for version in store.search_skill_versions(
            "reviewer",
            organization="acme",
            filter_string="version >= 2 AND tags.release = 'canary'",
        )
    ] == [2]


def test_search_skill_versions_orders_and_paginates_with_query_bound_tokens(store):
    for _ in range(3):
        store.create_skill_version("reviewer", organization="acme")
    store.create_skill_version("other", organization="acme")

    first_page = store.search_skill_versions(
        "reviewer",
        organization="acme",
        order_by=["version DESC"],
        max_results=2,
    )
    assert [version.version for version in first_page] == [3, 2]
    assert first_page.token is not None

    decoded = SkillRegistryPaginationToken.decode(first_page.token)
    assert decoded.query_scope == "skill_versions:acme/reviewer"
    assert decoded.offset == 2

    second_page = store.search_skill_versions(
        "reviewer",
        organization="acme",
        order_by=["version DESC"],
        max_results=2,
        page_token=first_page.token,
    )
    assert [version.version for version in second_page] == [1]
    assert second_page.token is None

    with pytest.raises(MlflowException, match="different query scope"):
        store.search_skill_versions(
            "other",
            organization="acme",
            order_by=["version DESC"],
            page_token=first_page.token,
        )


@pytest.mark.parametrize(
    "max_results",
    [0, -1, 1001, True, "1"],
)
def test_search_skill_versions_rejects_invalid_max_results(store, max_results):
    with pytest.raises(MlflowException, match="max_results") as exc:
        store.search_skill_versions("reviewer", max_results=max_results)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_skill_version_tags_can_be_set_updated_deleted_and_filtered(store):
    store.create_skill_version("reviewer", organization="acme", status=SkillStatus.DRAFT.value)

    store.set_skill_version_tag("reviewer", 1, "release", "canary", organization="acme")
    assert store.get_skill_version("reviewer", 1, organization="acme").tags == {
        "release": "canary"
    }

    store.set_skill_version_tag("reviewer", 1, "release", "stable", organization="acme")
    assert store.get_skill_version("reviewer", 1, organization="acme").tags == {
        "release": "stable"
    }
    assert [
        version.version
        for version in store.search_skill_versions(
            "reviewer",
            organization="acme",
            filter_string="tags.release = 'stable'",
        )
    ] == [1]

    store.delete_skill_version_tag("reviewer", 1, "release", organization="acme")
    assert store.get_skill_version("reviewer", 1, organization="acme").tags == {}

    with pytest.raises(MlflowException, match="Tag 'release' not found"):
        store.delete_skill_version_tag("reviewer", 1, "release", organization="acme")


def test_skill_version_tag_operations_reject_missing_or_deleted_versions(store):
    with pytest.raises(MlflowException, match="not found"):
        store.set_skill_version_tag("reviewer", 1, "release", "canary", organization="acme")

    store.create_skill_version("reviewer", organization="acme", status=SkillStatus.DRAFT.value)
    store.delete_skill_version("reviewer", 1, organization="acme")

    with pytest.raises(MlflowException, match="not found"):
        store.set_skill_version_tag("reviewer", 1, "release", "canary", organization="acme")
    with pytest.raises(MlflowException, match="not found"):
        store.delete_skill_version_tag("reviewer", 1, "release", organization="acme")
