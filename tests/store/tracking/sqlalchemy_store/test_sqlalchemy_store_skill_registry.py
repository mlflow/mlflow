from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Event
from types import SimpleNamespace
from unittest import mock

import pytest
import sqlalchemy
from sqlalchemy.dialects import mssql, mysql
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
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_CONFLICT,
    TEMPORARILY_UNAVAILABLE,
    ErrorCode,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlAgentPluginTag,
    SqlAgentPluginVersionTag,
    SqlSkill,
    SqlSkillAlias,
    SqlSkillTag,
    SqlSkillVersion,
    SqlSkillVersionTag,
)
from mlflow.store.tracking.skill_registry.sqlalchemy_mixin import (
    SqlAlchemySkillRegistryMixin,
    _skill_identity_predicate,
)
from mlflow.store.tracking.skill_registry_pagination import SkillRegistryPaginationToken
from mlflow.store.tracking.sqlalchemy_store import _DB_WRITE_MAX_DEADLOCK_RETRIES
from mlflow.utils.validation import (
    MAX_MODEL_REGISTRY_TAG_KEY_LENGTH,
    MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH,
)
from mlflow.utils.workspace_context import WorkspaceContext

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture
def mock_icon_hostname_resolution():
    with mock.patch(
        "mlflow.utils.validation._resolve_hostname_with_timeout",
        return_value=[(None, None, None, None, ("8.8.8.8", 0))],
    ):
        yield


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


def test_search_skills_filters_qualified_identities_before_pagination(store):
    store.create_skill("reviewer")
    store.create_skill("reviewer", organization="acme")
    store.create_skill("reviewer", organization="example")
    store.create_skill("writer")

    allowed = [("acme", "reviewer"), ("", "writer")]
    first = store.search_skills(max_results=1, include_skill_identities=allowed)
    second = store.search_skills(
        max_results=1, page_token=first.token, include_skill_identities=allowed
    )

    assert [(skill.organization, skill.name) for skill in first] == [("", "writer")]
    assert [(skill.organization, skill.name) for skill in second] == [("acme", "reviewer")]
    assert second.token is None
    assert list(store.search_skills(include_skill_identities=[])) == []
    assert [
        (skill.organization, skill.name)
        for skill in store.search_skills(exclude_skill_identities=[("acme", "reviewer")])
    ] == [("", "reviewer"), ("", "writer"), ("example", "reviewer")]

    changed = store.search_skills(
        max_results=1,
        page_token=first.token,
        include_skill_identities=[("example", "reviewer")],
    )
    assert list(changed) == []
    assert changed.token is None

    many_allowed = [("acme", f"missing-{index}") for index in range(500)] + allowed
    assert [
        (skill.organization, skill.name)
        for skill in store.search_skills(include_skill_identities=many_allowed)
    ] == [("", "writer"), ("acme", "reviewer")]

    # Exercise all predicates in the database, including SQL Server's NOT EXISTS
    # OPENJSON branch, with same-name Skills spanning several organizations.
    many_selected = many_allowed + [("", "reviewer"), ("example", "reviewer")]
    many_excluded = [("acme", f"excluded-{index}") for index in range(500)] + [
        ("", "reviewer"),
        ("", "writer"),
    ]
    query = {
        "include_skill_identities": many_selected,
        "exclude_skill_identities": many_excluded,
        "max_results": 1,
    }
    first = store.search_skills(**query)
    assert [(skill.organization, skill.name) for skill in first] == [("acme", "reviewer")]
    assert first.token is not None
    second = store.search_skills(**query, page_token=first.token)
    assert [(skill.organization, skill.name) for skill in second] == [("example", "reviewer")]
    assert second.token is None


@pytest.mark.parametrize("size", [300, 301, 2500])
def test_large_skill_identity_scope_uses_bounded_sql_server_parameters(size):
    identities = [(f"org-{index}", "reviewer") for index in range(size)]
    query = sqlalchemy.select(SqlSkill.name).where(
        _skill_identity_predicate(identities, "mssql"),
        ~_skill_identity_predicate(identities, "mssql"),
    )
    compiled = query.compile(dialect=mssql.dialect(), compile_kwargs={"render_postcompile": True})

    assert ("OPENJSON" in str(compiled)) is (size > 300)
    assert len(compiled.params) < 2100


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


def test_update_skill_preserves_resolved_fields(store, mock_icon_hostname_resolution):
    store.create_skill_version("reviewer")
    store.create_skill_version("reviewer", status=SkillStatus.DRAFT.value)

    updated = store.update_skill("reviewer", description="Updated")
    assert updated.latest_version == 1
    assert updated.status == SkillStatus.ACTIVE

    updated = store.update_skill("reviewer", icons=[{"src": "https://example.com/reviewer.svg"}])
    assert updated.latest_version == 1
    assert updated.status == SkillStatus.ACTIVE


def test_skill_source_type_is_resolved_from_latest_live_version(store):
    store.create_skill_version(
        "reviewer",
        source_type=SkillSourceType.GIT,
        source="https://example.com/reviewer.git",
    )
    store.create_skill_version(
        "reviewer",
        source_type=SkillSourceType.ZIP,
        source="https://example.com/reviewer.zip",
        status=SkillStatus.DRAFT.value,
    )

    skill = store.get_skill("reviewer")
    assert skill.latest_version == 1
    assert skill.source_type == SkillSourceType.GIT

    store.update_skill_version("reviewer", 1, status=SkillStatus.DEPRECATED.value)
    skill = store.get_skill("reviewer")
    assert skill.latest_version == 2
    assert skill.source_type == SkillSourceType.ZIP

    store.delete_skill_version("reviewer", 2)
    skill = store.get_skill("reviewer")
    assert skill.latest_version == 1
    assert skill.source_type == SkillSourceType.GIT

    store.delete_skill_version("reviewer", 1)
    skill = store.get_skill("reviewer")
    assert skill.latest_version is None
    assert skill.source_type is None


def test_skill_icons_round_trip_and_can_be_cleared(store, mock_icon_hostname_resolution):
    icons = [{"src": "https://example.com/reviewer.svg", "sizes": ["any"]}]
    created = store.create_skill("reviewer", icons=icons)
    assert created.icons == icons

    updated = store.update_skill("reviewer", icons=None)
    assert updated.icons is None

    unchanged = store.update_skill("reviewer", description="Updated")
    assert unchanged.icons is None


@pytest.mark.parametrize(
    ("icons", "message"),
    [
        ([{"src": "javascript:alert(1)"}], "Invalid Icon URL scheme"),
        (
            [{"src": "https://8.8.8.8/icon.png", "mimeType": "text/plain"}],
            "Invalid icon mimeType",
        ),
        ([{"src": "https://8.8.8.8/icon.png"}] * 101, "at most 100 items"),
    ],
)
def test_skill_icons_are_validated_on_create(store, icons, message, mock_icon_hostname_resolution):
    with pytest.raises(MlflowException, match=message) as exc:
        store.create_skill("reviewer", icons=icons)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    with pytest.raises(MlflowException, match="not found"):
        store.get_skill("reviewer")


def test_skill_icons_are_validated_on_update(store, mock_icon_hostname_resolution):
    original_icons = [{"src": "https://8.8.8.8/original.png"}]
    store.create_skill("reviewer", icons=original_icons)

    with pytest.raises(MlflowException, match="Invalid Icon URL scheme"):
        store.update_skill("reviewer", icons=[{"src": "javascript:alert(1)"}])

    assert store.get_skill("reviewer").icons == original_icons


def _get_skill_search_text(store, name="reviewer", organization="acme"):
    with store.ManagedSessionMaker() as session:
        return (
            store
            ._get_query(session, SqlSkill)
            .filter(SqlSkill.name == name, SqlSkill.organization == organization)
            .one()
            .search_text
        )


def test_skill_search_text_is_persisted_and_recomputed_for_description_only(
    store, mock_icon_hostname_resolution
):
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
    assert decoded.query_scope == "workspace:default:skills"
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


@pytest.mark.parametrize("target", ["parent", "version"])
@pytest.mark.parametrize("next_workspace", ["team-a", "team-b"])
def test_search_tokens_are_bound_to_workspace(store, workspaces_enabled, target, next_workspace):
    if not workspaces_enabled:
        pytest.skip("Workspace token binding is only applicable when workspaces are enabled")

    for workspace in ["team-a", "team-b"]:
        with WorkspaceContext(workspace):
            store.create_skill_version("alpha")
            store.create_skill_version("alpha")
            store.create_skill_version("beta")

    if target == "parent":
        search = store.search_skills
        attribute = "name"
        first_value = "alpha"
        second_value = "beta"
    else:

        def search(**kwargs):
            return store.search_skill_versions("alpha", **kwargs)

        attribute = "version"
        first_value = 1
        second_value = 2

    with WorkspaceContext("team-a"):
        first_page = search(max_results=1)
        assert [getattr(item, attribute) for item in first_page] == [first_value]
        assert first_page.token is not None

    with WorkspaceContext(next_workspace):
        if next_workspace == "team-a":
            second_page = search(max_results=1, page_token=first_page.token)
            assert [getattr(item, attribute) for item in second_page] == [second_value]
            assert second_page.token is None
        else:
            fresh_page = search(max_results=1)
            assert [getattr(item, attribute) for item in fresh_page] == [first_value]
            with pytest.raises(MlflowException, match="different query scope") as exc:
                search(max_results=1, page_token=first_page.token)
            assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_search_skills_filters_by_derived_status_organization_tags_and_search_text(store):
    store.create_skill("reviewer", organization="acme", description="Reviews pull requests")
    store.create_skill_version("reviewer", organization="acme")
    store.create_skill_version("reviewer", organization="acme", status=SkillStatus.DRAFT.value)
    store.set_skill_tag("reviewer", "team", "platform", organization="acme")
    store.set_skill_tag("reviewer", "priority", "high", organization="acme")

    store.create_skill("draft-helper", organization="acme", description="Draft assistance")
    store.create_skill_version("draft-helper", organization="acme", status=SkillStatus.DRAFT.value)
    store.set_skill_tag("draft-helper", "team", "platform", organization="acme")

    store.create_skill("reviewer", organization="beta", description="Reviews beta code")
    store.create_skill_version("reviewer", organization="beta")
    store.set_skill_tag("reviewer", "team", "platform", organization="beta")

    active_acme_platform = store.search_skills(
        filter_string=("organization = 'acme' AND status = 'active' AND tags.team = 'platform'")
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
    assert [(skill.organization, skill.name) for skill in text_matches] == [("acme", "reviewer")]


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


def test_search_skills_reuses_joined_latest_version_columns(store):
    store.create_skill_version(
        "git-reviewer",
        organization="acme",
        source_type=SkillSourceType.GIT,
        source="https://example.com/skills.git",
    )

    statements = []

    def capture_statement(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    sqlalchemy.event.listen(store.engine, "before_cursor_execute", capture_statement)
    try:
        results = store.search_skills(
            filter_string="organization = 'acme' AND status = 'active' AND source_type = 'git'",
            order_by=["source_type ASC", "status ASC"],
        )
    finally:
        sqlalchemy.event.remove(store.engine, "before_cursor_execute", capture_statement)

    assert [skill.name for skill in results] == ["git-reviewer"]
    sql = statements[0].lower()
    assert "skill_latest_candidates" in sql
    assert "resolved_skill_status_candidates" not in sql
    assert "resolved_skill_source_type_candidates" not in sql
    assert sql.count("row_number()") == 1


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

            query, _ = SqlAlchemySkillRegistryMixin._skill_query(owner, session)
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
    assert [skill.name for skill in store.search_skills("tags.team LIKE 'ml-%'")] == ["reviewer"]

    store.delete_skill_tag("reviewer", "team", organization="acme")
    assert store.get_skill("reviewer", organization="acme").tags == {}

    with pytest.raises(MlflowException, match="Tag 'team' not found"):
        store.delete_skill_tag("reviewer", "team", organization="acme")


@pytest.mark.parametrize("key", [None, "k" * (MAX_MODEL_REGISTRY_TAG_KEY_LENGTH + 1)])
def test_delete_skill_tag_rejects_invalid_key(store, key):
    store.create_skill("reviewer", organization="acme")

    with pytest.raises(
        MlflowException,
        match="Missing value for required parameter|exceeds the maximum length",
    ) as exc:
        store.delete_skill_tag("reviewer", key, organization="acme")

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_skill_tag_keys_are_case_sensitive(store):
    store.create_skill("reviewer", organization="acme")
    store.set_skill_tag("reviewer", "Team", "original", organization="acme")
    store.set_skill_tag("reviewer", "team", "new", organization="acme")

    assert store.get_skill("reviewer", organization="acme").tags == {
        "Team": "original",
        "team": "new",
    }
    assert [skill.name for skill in store.search_skills("tags.team = 'new'")] == ["reviewer"]

    store.delete_skill_tag("reviewer", "team", organization="acme")
    assert store.get_skill("reviewer", organization="acme").tags == {"Team": "original"}


def test_skill_tag_value_uses_registry_limits_without_truncation(store):
    store.create_skill("reviewer", organization="acme")
    value = "x" * 9000

    store.set_skill_tag("reviewer", "team", value, organization="acme")

    assert store.get_skill("reviewer", organization="acme").tags["team"] == value

    with pytest.raises(MlflowException, match="exceeds the maximum length"):
        store.set_skill_tag(
            "reviewer",
            "team",
            "x" * (MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH + 1),
            organization="acme",
        )


@pytest.mark.parametrize(
    "tag_model",
    [SqlSkillTag, SqlSkillVersionTag, SqlAgentPluginTag, SqlAgentPluginVersionTag],
)
@pytest.mark.parametrize(
    ("dialect", "expected_collation"),
    [
        (mysql.dialect(), "utf8mb4_bin"),
        (mssql.dialect(), "SQL_Latin1_General_CP1_CS_AS"),
    ],
)
def test_skill_registry_tag_key_columns_are_case_sensitive(tag_model, dialect, expected_collation):
    tag_key_type = tag_model.__table__.c.key.type.dialect_impl(dialect)
    assert tag_key_type.collation == expected_collation


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        (mysql.dialect(), "MEDIUMTEXT"),
        (mssql.dialect(), "NVARCHAR(max)"),
    ],
)
@pytest.mark.parametrize(
    "tag_model",
    [SqlSkillTag, SqlSkillVersionTag, SqlAgentPluginTag, SqlAgentPluginVersionTag],
)
def test_skill_registry_tag_value_columns_use_backend_text_type(tag_model, dialect, expected):
    tag_value_type = tag_model.__table__.c.value.type
    assert tag_value_type.compile(dialect=dialect) == expected


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


def test_skill_tag_search_is_workspace_scoped(store, workspaces_enabled):
    if not workspaces_enabled:
        pytest.skip("Workspace isolation is only applicable when workspaces are enabled")

    with WorkspaceContext("team-a"):
        store.create_skill("reviewer", organization="acme")
        store.set_skill_tag("reviewer", "team", "platform", organization="acme")

    with WorkspaceContext("team-b"):
        store.create_skill("reviewer", organization="acme")
        assert store.get_skill("reviewer", organization="acme").tags == {}
        assert store.search_skills(filter_string="tags.team = 'platform'") == []

    with WorkspaceContext("team-a"):
        results = store.search_skills(filter_string="tags.team = 'platform'")
        assert [skill.workspace for skill in results] == ["team-a"]


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


def test_skill_version_auto_creates_parent_and_preserves_existing_parent(
    store, mock_icon_hostname_resolution
):
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


def _bulk_definition(name="reviewer", **overrides):
    return {
        "name": name,
        "source_type": "git",
        "source": "https://example.com/skills.git",
        "ref": "main",
        "subpath": f"skills/{name}",
        "digest": "a" * 64,
        **overrides,
    }


@pytest.mark.parametrize("status", ["active", "draft"])
def test_bulk_register_skills_preserves_order_inputs_and_existing_metadata(store, status):
    store.create_skill("reviewer", organization="acme", description="Keep", created_by="alice")
    existing = store.create_skill_version(
        **_bulk_definition(), organization="acme", created_by="alice"
    )
    batch = [_bulk_definition("writer", status=status), _bulk_definition(status=status)]
    original = deepcopy(batch)
    parent = store.get_skill("reviewer", organization="acme")

    result = store.bulk_register_skills(batch, organization="acme", created_by="bob")

    assert [(v.name, v.version) for v in result] == [("writer", 1), ("reviewer", 1)]
    assert result[0].created_by == "bob"
    assert result[0].status == status
    assert result[1] == existing
    assert store.get_skill("reviewer", organization="acme") == parent
    assert store.get_skill("writer", organization="acme").created_by == "bob"
    assert store.get_skill("writer", organization="acme").description is None
    assert batch == original
    assert store.bulk_register_skills(batch, organization="acme") == result
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkillVersion).count() == 2
    # Ordinary registration still creates a fresh version after an identical bulk import.
    assert store.create_skill_version(**_bulk_definition(), organization="acme").version == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "https://example.com/other.git"},
        {"source": "https://example.com/Skills.git"},
        {"ref": "release"},
        {"ref": "Main"},
        {"subpath": "other/reviewer"},
        {"subpath": "skills/Reviewer"},
        {"digest": "b" * 64},
    ],
)
def test_bulk_register_skills_new_version_for_changed_definition(store, changes):
    store.bulk_register_skills([_bulk_definition()])
    changed = _bulk_definition(**changes)
    assert store.bulk_register_skills([changed])[0].version == 2
    assert store.bulk_register_skills([changed])[0].version == 2


@pytest.mark.parametrize(
    ("first_subpath", "second_subpath"),
    [
        ("skills/a/", "skills/a"),
        ("skills/a", "skills/a/"),
        (None, ""),
        ("", None),
    ],
)
def test_bulk_register_skills_reuses_equivalent_subpaths(store, first_subpath, second_subpath):
    original = store.bulk_register_skills(
        [_bulk_definition(subpath=first_subpath)], created_by="original"
    )[0]

    repeated = store.bulk_register_skills(
        [_bulk_definition(subpath=second_subpath)], created_by="importer"
    )[0]

    assert repeated.version == original.version
    assert repeated == original
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkillVersion).count() == 1


@pytest.mark.parametrize("status", ["active", "draft", "deprecated"])
@pytest.mark.parametrize("requested_status", ["active", "draft"])
def test_bulk_register_skills_reuses_highest_non_deleted_match(store, status, requested_status):
    for _ in range(3):
        store.create_skill_version(**_bulk_definition(), created_by="original")
    with store.ManagedSessionMaker(read_only=False) as session:
        versions = store._get_query(session, SqlSkillVersion)
        versions.filter(SqlSkillVersion.version == 2).update({SqlSkillVersion.status: status})
        versions.filter(SqlSkillVersion.version == 3).update({SqlSkillVersion.status: "deleted"})
    existing = store.get_skill_version("reviewer", 2)
    result = store.bulk_register_skills(
        [_bulk_definition(status=requested_status)], created_by="importer"
    )[0]
    assert result == existing
    assert result.version == 2
    assert result.status == status
    assert result.created_by == "original"
    assert result.last_updated_by == "original"
    assert store.get_skill_version("reviewer", 2) == existing
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkillVersion).count() == 3


@pytest.mark.parametrize("status", ["deprecated", "deleted", "invalid", None])
def test_bulk_register_skills_invalid_status_does_not_write(store, status):
    with pytest.raises(MlflowException, match="status") as exc:
        store.bulk_register_skills([
            _bulk_definition(),
            _bulk_definition("writer", status=status),
        ])
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkill).count() == 0
        assert store._get_query(session, SqlSkillVersion).count() == 0


@pytest.mark.parametrize("status", ["active", "draft"])
def test_bulk_register_skills_keeps_deleted_history_and_restarts_after_hard_delete(store, status):
    store.bulk_register_skills([_bulk_definition()])
    with store.ManagedSessionMaker(read_only=False) as session:
        store._get_query(session, SqlSkillVersion).update({SqlSkillVersion.status: "deleted"})
    imported = store.bulk_register_skills([_bulk_definition(status=status)])[0]
    assert imported.version == 2
    assert imported.status == status
    with store.ManagedSessionMaker() as session:
        deleted = store._get_query(session, SqlSkillVersion).filter_by(version=1).one()
        assert deleted.status == "deleted"
    assert store.bulk_register_skills([_bulk_definition(status=status)])[0] == imported
    store.delete_skill("reviewer")
    recreated = store.bulk_register_skills([_bulk_definition(status=status)])[0]
    assert recreated.version == 1
    assert recreated.status == status


@pytest.mark.parametrize(
    "batch",
    [
        [],
        None,
        [None],
        [_bulk_definition(name=None)],
        [_bulk_definition(name="UPPER")],
        [_bulk_definition(digest=None)],
        [_bulk_definition(digest="invalid")],
        [_bulk_definition(source=None)],
        [_bulk_definition(source_type="oci")],
        [_bulk_definition(), _bulk_definition()],
        [_bulk_definition(), _bulk_definition("writer", ref="other")],
        [_bulk_definition(), _bulk_definition("writer", source="https://example.com/other.git")],
        [_bulk_definition(), _bulk_definition("writer", status="draft")],
        [_bulk_definition(status="draft"), _bulk_definition("writer")],
    ],
)
def test_bulk_register_skills_invalid_batch_does_not_write(store, batch):
    with pytest.raises(MlflowException, match="Skill|skill|digest") as exc:
        store.bulk_register_skills(batch)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkill).count() == 0
        assert store._get_query(session, SqlSkillVersion).count() == 0


def test_bulk_register_skills_later_failure_rolls_back_every_new_parent_and_version(store):
    persist = store._persist_skill_version

    def fail_after_second_insert(*args, **kwargs):
        result = persist(*args, **kwargs)
        if kwargs["name"] == "writer":
            raise MlflowException.invalid_parameter_value("Injected later failure")
        return result

    with mock.patch.object(
        store, "_persist_skill_version", side_effect=fail_after_second_insert
    ) as mock_persist:
        with pytest.raises(MlflowException, match="Injected later failure"):
            store.bulk_register_skills([_bulk_definition(), _bulk_definition("writer")])
    assert mock_persist.call_count == 2
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkill).count() == 0
        assert store._get_query(session, SqlSkillVersion).count() == 0


def test_bulk_register_skills_rejects_changed_parent_state_before_writing(store):
    store.create_skill("writer", created_by="owner")

    with pytest.raises(
        MlflowException, match="already exists; this registration expected to create it"
    ) as exc:
        store.bulk_register_skills(
            [_bulk_definition(), _bulk_definition("writer")],
            expected_parent_exists={"reviewer": False, "writer": False},
        )

    assert exc.value.error_code == ErrorCode.Name(RESOURCE_CONFLICT)
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkill).count() == 1
        assert store._get_query(session, SqlSkillVersion).count() == 0


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("expected_parent_exists", [False, True])
def test_registration_conflict_describes_parent_state(store, bulk, expected_parent_exists):
    if not expected_parent_exists:
        store.create_skill("reviewer")
    message = (
        "no longer exists; this registration expected to update it"
        if expected_parent_exists
        else "already exists; this registration expected to create it"
    )
    register = store.bulk_register_skills if bulk else store.create_skill_version
    args = [_bulk_definition()] if bulk else "reviewer"
    expectation = {"reviewer": expected_parent_exists} if bulk else expected_parent_exists
    with pytest.raises(MlflowException, match=message) as exc:
        register(args, expected_parent_exists=expectation)
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_CONFLICT)
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkillVersion).count() == 0


@pytest.mark.parametrize("exhaust", [False, True])
def test_bulk_register_skills_retries_creation_conflict_as_whole_transaction(store, exhaust):
    persist = store._persist_skill_version
    calls = 0

    def conflict_after_second_insert(*args, **kwargs):
        nonlocal calls
        result = persist(*args, **kwargs)
        if kwargs["name"] == "writer":
            calls += 1
            if calls == 1 or exhaust:
                raise MlflowException("allocation collision", RESOURCE_ALREADY_EXISTS) from (
                    sqlalchemy.exc.IntegrityError("insert", {}, Exception("duplicate key"))
                )
        return result

    with mock.patch.object(
        store, "_persist_skill_version", side_effect=conflict_after_second_insert
    ) as mock_persist:
        if exhaust:
            with pytest.raises(MlflowException, match="allocation collision"):
                store.bulk_register_skills([_bulk_definition(), _bulk_definition("writer")])
        else:
            result = store.bulk_register_skills([_bulk_definition(), _bulk_definition("writer")])
            assert [v.version for v in result] == [1, 1]
    assert calls == (store.CREATE_SKILL_VERSION_RETRIES if exhaust else 2)
    assert mock_persist.call_count == 2 * calls
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkillVersion).count() == (0 if exhaust else 2)
        assert store._get_query(session, SqlSkill).count() == (0 if exhaust else 2)


@pytest.mark.parametrize("bulk", [False, True], ids=["standalone", "bulk"])
def test_skill_registration_retries_deadlock_with_a_fresh_transaction(store, bulk):
    persist = store._persist_skill_version
    sessions = []

    def deadlock_after_insert(*args, **kwargs):
        sessions.append(kwargs["session"])
        result = persist(*args, **kwargs)
        if len(sessions) == 1:
            raise MlflowException("deadlock victim", TEMPORARILY_UNAVAILABLE)
        return result

    with (
        mock.patch.object(
            store, "_persist_skill_version", side_effect=deadlock_after_insert
        ) as mock_persist,
        mock.patch("mlflow.store.tracking.sqlalchemy_store.time.sleep") as mock_sleep,
    ):
        if bulk:
            result = store.bulk_register_skills([_bulk_definition()])[0]
        else:
            result = store.create_skill_version(**_bulk_definition())
        assert result.version == 1
    assert mock_persist.call_count == 2
    mock_sleep.assert_called_once()
    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]


@pytest.mark.parametrize(
    ("error_code", "message", "attempts"),
    [
        (TEMPORARILY_UNAVAILABLE, "deadlock victim", _DB_WRITE_MAX_DEADLOCK_RETRIES + 1),
        (TEMPORARILY_UNAVAILABLE, "connection unavailable", 1),
        (INVALID_PARAMETER_VALUE, "invalid input mentioning deadlock", 1),
    ],
)
def test_create_skill_version_failure_retries_are_bounded(store, error_code, message, attempts):
    persist = store._persist_skill_version
    sessions = []

    def fail_after_insert(*args, **kwargs):
        sessions.append(kwargs["session"])
        persist(*args, **kwargs)
        raise MlflowException(message, error_code)

    with (
        mock.patch.object(
            store, "_persist_skill_version", side_effect=fail_after_insert
        ) as mock_persist,
        mock.patch("mlflow.store.tracking.sqlalchemy_store.time.sleep") as mock_sleep,
        pytest.raises(MlflowException, match=message) as exc,
    ):
        store.create_skill_version(**_bulk_definition())

    assert exc.value.error_code == ErrorCode.Name(error_code)
    assert mock_persist.call_count == attempts
    if attempts == 1:
        mock_sleep.assert_not_called()
    else:
        assert mock_sleep.call_count == attempts - 1
    assert len(sessions) == attempts
    assert len({id(session) for session in sessions}) == attempts
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkill).count() == 0
        assert store._get_query(session, SqlSkillVersion).count() == 0


def test_bulk_register_skills_organization_and_workspace_isolation(store, workspaces_enabled):
    for _ in range(2):
        store.create_skill_version(**_bulk_definition(), organization="first", created_by="alice")
    second = store.bulk_register_skills(
        [_bulk_definition()], organization="second", created_by="bob"
    )[0]
    first = store.bulk_register_skills([_bulk_definition()], organization="first")[0]
    assert (first.organization, first.version, first.created_by) == ("first", 2, "alice")
    assert (second.organization, second.version, second.created_by) == ("second", 1, "bob")
    assert store.get_skill_version("reviewer", 2, organization="first") == first
    assert store.get_skill_version("reviewer", 1, organization="second") == second
    changed = _bulk_definition(digest="b" * 64)
    for organization, version in [("first", 3), ("second", 2)]:
        result = store.bulk_register_skills([changed], organization=organization)[0]
        assert (result.organization, result.version) == (organization, version)
        assert store.get_skill_version("reviewer", version, organization=organization) == result
    if not workspaces_enabled:
        return
    with WorkspaceContext("team-a"):
        assert store.bulk_register_skills([_bulk_definition()])[0].version == 1
    with WorkspaceContext("team-b"):
        result = store.bulk_register_skills([_bulk_definition()])[0]
        assert result.version == 1
        assert result.workspace == "team-b"
    with WorkspaceContext("team-a"):
        assert store.bulk_register_skills([_bulk_definition()])[0].version == 1


@pytest.mark.parametrize("parents_exist", [False, True])
def test_bulk_register_skills_concurrent_overlapping_batches(store, parents_exist):
    if parents_exist:
        store.create_skill("reviewer")
        store.create_skill("writer")
    start = Barrier(2)

    def register(batch):
        with WorkspaceContext("default"):
            start.wait(timeout=10)
            return store.bulk_register_skills(batch)

    first = [_bulk_definition("writer"), _bulk_definition()]
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="bulk-import") as executor:
        calls = [executor.submit(register, first), executor.submit(register, list(reversed(first)))]
        results = [call.result(timeout=30) for call in calls]
    assert [(v.name, v.version) for v in results[0]] == [("writer", 1), ("reviewer", 1)]
    assert [(v.name, v.version) for v in results[1]] == [("reviewer", 1), ("writer", 1)]
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkillVersion).count() == 2


def test_bulk_register_skills_overlaps_an_already_allocated_single_registration(store):
    store.create_skill("reviewer")
    allocated = Event()
    resume = Event()
    attempted_versions = []
    persist = store._persist_skill_version

    def pause_first_allocation(*args, **kwargs):
        if kwargs.get("created_by") == "single":
            attempted_versions.append(kwargs["version"])
            if not allocated.is_set():
                allocated.set()
                assert resume.wait(timeout=10)
        return persist(*args, **kwargs)

    def register_single():
        with WorkspaceContext("default"):
            return store.create_skill_version(**_bulk_definition(), created_by="single")

    with (
        mock.patch.object(
            store, "_persist_skill_version", side_effect=pause_first_allocation
        ) as mock_persist,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="single-before-bulk") as executor,
    ):
        single = executor.submit(register_single)
        try:
            assert allocated.wait(timeout=10)
            bulk = store.bulk_register_skills([_bulk_definition()], created_by="bulk")[0]
        finally:
            resume.set()
        registered = single.result(timeout=10)

    assert mock_persist.call_count == 3
    assert attempted_versions == [1, 2]
    assert (bulk.version, registered.version) == (1, 2)
    assert store.get_skill_version("reviewer", 1).created_by == "bulk"
    assert store.get_skill_version("reviewer", 2).created_by == "single"
    assert store.bulk_register_skills([_bulk_definition()])[0] == registered


@pytest.mark.parametrize("operation", ["single-registration", "deletion"])
def test_bulk_register_skills_final_state_with_concurrent_writer(store, operation):
    store.create_skill("reviewer")
    started = Event()
    persist = store._persist_skill_version

    def other_write():
        with WorkspaceContext("default"):
            started.set()
            if operation == "deletion":
                return store.delete_skill("reviewer")
            return store.create_skill_version(**_bulk_definition(), created_by="single")

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="concurrent-skill-write") as executor:
        future = None

        def persist_and_start_other_writer(*args, **kwargs):
            nonlocal future
            if kwargs.get("created_by") == "bulk":
                result = persist(*args, **kwargs)
                future = executor.submit(other_write)
                assert started.wait(timeout=10)
                return result
            return persist(*args, **kwargs)

        with mock.patch.object(
            store, "_persist_skill_version", side_effect=persist_and_start_other_writer
        ) as mock_persist:
            assert (
                store.bulk_register_skills([_bulk_definition()], created_by="bulk")[0].version == 1
            )
            result = future.result(timeout=30)
    if operation == "single-registration":
        assert mock_persist.call_count == 2
        assert result.version == 2
        assert store.get_skill_version("reviewer", 1).created_by == "bulk"
        assert store.get_skill_version("reviewer", 2) == result
    else:
        mock_persist.assert_called_once()
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkill).count() == (0 if operation == "deletion" else 1)
        assert store._get_query(session, SqlSkillVersion).count() == (
            0 if operation == "deletion" else 2
        )


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
    assert decoded.query_scope == "workspace:default:skill_versions:acme/reviewer"
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
    assert store.get_skill_version("reviewer", 1, organization="acme").tags == {"release": "canary"}

    store.set_skill_version_tag("reviewer", 1, "release", "stable", organization="acme")
    assert store.get_skill_version("reviewer", 1, organization="acme").tags == {"release": "stable"}
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


@pytest.mark.parametrize("key", [None, "k" * (MAX_MODEL_REGISTRY_TAG_KEY_LENGTH + 1)])
def test_delete_skill_version_tag_rejects_invalid_key(store, key):
    store.create_skill_version("reviewer", organization="acme", status=SkillStatus.DRAFT.value)

    with pytest.raises(
        MlflowException,
        match="Missing value for required parameter|exceeds the maximum length",
    ) as exc:
        store.delete_skill_version_tag("reviewer", 1, key, organization="acme")

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_skill_version_tag_keys_are_case_sensitive(store):
    store.create_skill_version("reviewer", organization="acme", status=SkillStatus.DRAFT.value)
    store.set_skill_version_tag("reviewer", 1, "Team", "original", organization="acme")
    store.set_skill_version_tag("reviewer", 1, "team", "new", organization="acme")

    assert store.get_skill_version("reviewer", 1, organization="acme").tags == {
        "Team": "original",
        "team": "new",
    }
    assert [
        version.version
        for version in store.search_skill_versions(
            "reviewer",
            organization="acme",
            filter_string="tags.team = 'new'",
        )
    ] == [1]

    store.delete_skill_version_tag("reviewer", 1, "team", organization="acme")
    assert store.get_skill_version("reviewer", 1, organization="acme").tags == {"Team": "original"}


def test_skill_version_tag_value_uses_registry_limits_without_truncation(store):
    store.create_skill_version("reviewer", organization="acme", status=SkillStatus.DRAFT.value)
    value = "x" * 9000

    store.set_skill_version_tag("reviewer", 1, "release", value, organization="acme")

    assert store.get_skill_version("reviewer", 1, organization="acme").tags["release"] == value

    with pytest.raises(MlflowException, match="exceeds the maximum length"):
        store.set_skill_version_tag(
            "reviewer",
            1,
            "release",
            "x" * (MAX_MODEL_REGISTRY_TAG_VALUE_LENGTH + 1),
            organization="acme",
        )


def test_skill_version_tag_operations_reject_missing_or_deleted_versions(store):
    with pytest.raises(MlflowException, match="not found"):
        store.set_skill_version_tag("reviewer", 1, "release", "canary", organization="acme")

    store.create_skill_version("reviewer", organization="acme", status=SkillStatus.DRAFT.value)
    store.delete_skill_version("reviewer", 1, organization="acme")

    with pytest.raises(MlflowException, match="not found"):
        store.set_skill_version_tag("reviewer", 1, "release", "canary", organization="acme")
    with pytest.raises(MlflowException, match="not found"):
        store.delete_skill_version_tag("reviewer", 1, "release", organization="acme")
