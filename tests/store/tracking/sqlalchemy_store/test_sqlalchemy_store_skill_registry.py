from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Event
from types import SimpleNamespace
from unittest import mock

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
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    TEMPORARILY_UNAVAILABLE,
    ErrorCode,
)
from mlflow.store.tracking.dbmodels.models import (
    SqlSkill,
    SqlSkillAlias,
    SqlSkillTag,
    SqlSkillVersion,
)
from mlflow.store.tracking.skill_registry.sqlalchemy_mixin import SqlAlchemySkillRegistryMixin
from mlflow.store.tracking.sqlalchemy_store import _DB_WRITE_MAX_DEADLOCK_RETRIES
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


def test_bulk_register_skills_preserves_order_inputs_and_existing_metadata(store):
    store.create_skill("reviewer", organization="acme", description="Keep", created_by="alice")
    existing = store.create_skill_version(
        **_bulk_definition(), organization="acme", created_by="alice"
    )
    batch = [_bulk_definition("writer"), _bulk_definition()]
    original = deepcopy(batch)
    parent = store.get_skill("reviewer", organization="acme")

    result = store.bulk_register_skills(batch, organization="acme", created_by="bob")

    assert [(v.name, v.version) for v in result] == [("writer", 1), ("reviewer", 1)]
    assert result[0].created_by == "bob"
    assert result[0].status == SkillStatus.ACTIVE
    assert result[1] == existing
    assert store.get_skill("reviewer", organization="acme") == parent
    assert store.get_skill("writer", organization="acme").created_by == "bob"
    assert store.get_skill("writer", organization="acme").description is None
    assert batch == original
    assert [v.version for v in store.bulk_register_skills(batch, organization="acme")] == [1, 1]
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


@pytest.mark.parametrize("status", ["active", "draft", "deprecated"])
def test_bulk_register_skills_reuses_highest_active_match(store, status):
    for _ in range(3):
        store.create_skill_version(**_bulk_definition(), created_by="original")
    with store.ManagedSessionMaker(read_only=False) as session:
        versions = store._get_query(session, SqlSkillVersion)
        versions.filter(SqlSkillVersion.version == 2).update({SqlSkillVersion.status: status})
        versions.filter(SqlSkillVersion.version == 3).update({SqlSkillVersion.status: "deleted"})
    result = store.bulk_register_skills([_bulk_definition()], created_by="importer")[0]
    assert result.version == (2 if status == "active" else 1)
    assert result.status == "active"
    assert result.created_by == "original"
    assert result.last_updated_by == "original"


@pytest.mark.parametrize("status", ["draft", "deprecated"])
def test_bulk_register_skills_creates_active_version_for_inactive_match(store, status):
    store.create_skill_version(**_bulk_definition(), created_by="original")
    with store.ManagedSessionMaker(read_only=False) as session:
        store._get_query(session, SqlSkillVersion).update({SqlSkillVersion.status: status})
    result = store.bulk_register_skills([_bulk_definition()], created_by="importer")[0]
    assert result.version == 2
    assert result.status == "active"
    assert result.created_by == "importer"
    assert store.get_skill_version("reviewer", 1).status == status
    assert store.bulk_register_skills([_bulk_definition()])[0].version == 2


def test_bulk_register_skills_keeps_deleted_history_and_restarts_after_hard_delete(store):
    store.bulk_register_skills([_bulk_definition()])
    with store.ManagedSessionMaker(read_only=False) as session:
        store._get_query(session, SqlSkillVersion).update({SqlSkillVersion.status: "deleted"})
    assert store.bulk_register_skills([_bulk_definition()])[0].version == 2
    with store.ManagedSessionMaker() as session:
        deleted = store._get_query(session, SqlSkillVersion).filter_by(version=1).one()
        assert deleted.status == "deleted"
    assert store.bulk_register_skills([_bulk_definition()])[0].version == 2
    store.delete_skill("reviewer")
    assert store.bulk_register_skills([_bulk_definition()])[0].version == 1


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

    with mock.patch.object(store, "_persist_skill_version", side_effect=fail_after_second_insert):
        with pytest.raises(MlflowException, match="Injected later failure"):
            store.bulk_register_skills([_bulk_definition(), _bulk_definition("writer")])
    with store.ManagedSessionMaker() as session:
        assert store._get_query(session, SqlSkill).count() == 0
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
    ):
        if exhaust:
            with pytest.raises(MlflowException, match="allocation collision"):
                store.bulk_register_skills([_bulk_definition(), _bulk_definition("writer")])
        else:
            result = store.bulk_register_skills([_bulk_definition(), _bulk_definition("writer")])
            assert [v.version for v in result] == [1, 1]
    assert calls == (store.CREATE_SKILL_VERSION_RETRIES if exhaust else 2)
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
        mock.patch.object(store, "_persist_skill_version", side_effect=deadlock_after_insert),
        mock.patch("mlflow.store.tracking.sqlalchemy_store.time.sleep"),
    ):
        if bulk:
            result = store.bulk_register_skills([_bulk_definition()])[0]
        else:
            result = store.create_skill_version(**_bulk_definition())
        assert result.version == 1
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
        mock.patch.object(store, "_persist_skill_version", side_effect=fail_after_insert),
        mock.patch("mlflow.store.tracking.sqlalchemy_store.time.sleep"),
        pytest.raises(MlflowException, match=message) as exc,
    ):
        store.create_skill_version(**_bulk_definition())

    assert exc.value.error_code == ErrorCode.Name(error_code)
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
        mock.patch.object(store, "_persist_skill_version", side_effect=pause_first_allocation),
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="single-before-bulk") as executor,
    ):
        single = executor.submit(register_single)
        try:
            assert allocated.wait(timeout=10)
            bulk = store.bulk_register_skills([_bulk_definition()], created_by="bulk")[0]
        finally:
            resume.set()
        registered = single.result(timeout=10)

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
        ):
            assert (
                store.bulk_register_skills([_bulk_definition()], created_by="bulk")[0].version == 1
            )
            result = future.result(timeout=30)
    if operation == "single-registration":
        assert result.version == 2
        assert store.get_skill_version("reviewer", 1).created_by == "bulk"
        assert store.get_skill_version("reviewer", 2) == result
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
