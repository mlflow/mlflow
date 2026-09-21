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
from mlflow.entities.workspace import Workspace
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.store.tracking.dbmodels.models import (
    SqlSkill,
    SqlSkillAlias,
    SqlSkillTag,
    SqlSkillVersion,
)
from mlflow.store.tracking.skill_registry.sqlalchemy_mixin import SqlAlchemySkillRegistryMixin
from mlflow.tracking._tracking_service.utils import _get_sqlalchemy_store
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


def test_search_skills_returns_stable_paginated_results(store):
    store.create_skill("writer", organization="acme")
    store.create_skill("reviewer", organization="acme")

    first_page = store.search_skills(max_results=1)
    assert [skill.name for skill in first_page] == ["reviewer"]
    assert first_page.token is not None

    second_page = store.search_skills(max_results=1, page_token=first_page.token)
    assert [skill.name for skill in second_page] == ["writer"]
    assert second_page.token is None


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


def test_cannot_disable_workspaces_with_non_default_skills(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    artifacts = str(tmp_path / "artifacts")

    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
    with WorkspaceContext("default"):
        scoped = _get_sqlalchemy_store(uri, artifacts)
        scoped._get_workspace_provider_instance().create_workspace(Workspace(name="private-team"))
    with WorkspaceContext("private-team"):
        scoped.create_skill("secret-skill", description="private data")
        scoped.create_skill_version("secret-skill")

    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")

    with pytest.raises(MlflowException, match="Skills exist outside the default workspace") as exc:
        _get_sqlalchemy_store(uri, artifacts)

    assert exc.value.error_code == "INVALID_STATE"


@pytest.mark.parametrize(
    "operation",
    ["get_skill", "search_skills", "update_skill", "get_skill_version", "create_skill_version"],
)
def test_existing_single_tenant_store_ignores_later_private_skills(
    tmp_path, monkeypatch, operation
):
    uri = f"sqlite:///{tmp_path / 'registry.db'}"
    artifacts = str(tmp_path / "artifacts")

    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")
    single_tenant = _get_sqlalchemy_store(uri, artifacts)
    single_tenant.create_skill("public-skill", organization="acme")

    with monkeypatch.context() as server_b:
        server_b.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
        with WorkspaceContext("default"):
            workspace_store = _get_sqlalchemy_store(uri, artifacts)
            workspace_store._get_workspace_provider_instance().create_workspace(
                Workspace(name="private-team")
            )
        with WorkspaceContext("private-team"):
            workspace_store.create_skill(
                "secret-skill", organization="acme", description="Private description"
            )
            for _ in range(2):
                workspace_store.create_skill_version("secret-skill", organization="acme")

    if operation == "search_skills":
        skills = single_tenant.search_skills()
        assert [(skill.name, skill.workspace) for skill in skills] == [("public-skill", "default")]
    elif operation == "create_skill_version":
        created = single_tenant.create_skill_version("secret-skill", organization="acme")
        assert (created.workspace, created.version) == ("default", 1)
    else:

        def access_private_skill():
            if operation == "get_skill":
                return single_tenant.get_skill("secret-skill", organization="acme")
            if operation == "get_skill_version":
                return single_tenant.get_skill_version("secret-skill", 1, organization="acme")
            return single_tenant.update_skill(
                "secret-skill", organization="acme", description="Changed by server A"
            )

        with pytest.raises(MlflowException, match="not found") as exc:
            access_private_skill()
        assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"

    with monkeypatch.context() as server_b:
        server_b.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
        with WorkspaceContext("private-team"):
            private = workspace_store.get_skill("secret-skill", organization="acme")
            assert private.description == "Private description"
            assert private.latest_version == 2


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
