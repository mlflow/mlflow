import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
import sqlalchemy

from mlflow.entities import SkillSourceType, SkillStatus
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import SqlSkill, SqlSkillAlias, SqlSkillVersion

pytestmark = pytest.mark.notrackingurimock


def _seed_skill(
    store, versions, *, name="reviewer", organization="", source_type=None, source=None
):
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
                    source_type=source_type,
                    source=source,
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


def _get_version_audit(store, version, *, name="reviewer", organization=""):
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
        return row.last_updated_by, row.last_updated_at


def _get_alias_rows(store, *, name="reviewer", organization=""):
    with store.ManagedSessionMaker() as session:
        return (
            store
            ._get_query(session, SqlSkillAlias)
            .with_entities(SqlSkillAlias.alias, SqlSkillAlias.version)
            .filter(
                SqlSkillAlias.name == name,
                SqlSkillAlias.organization == organization,
            )
            .all()
        )


@pytest.mark.parametrize(
    ("kwargs", "expected_status"),
    [({}, SkillStatus.ACTIVE), ({"status": SkillStatus.DRAFT}, SkillStatus.DRAFT)],
    ids=["default-active", "explicit-draft"],
)
def test_create_skill_version_registration_status(store, kwargs, expected_status):
    created = store.create_skill_version("reviewer", **kwargs)

    assert created.status == expected_status
    assert store.get_skill_version("reviewer", created.version) == created
    parent = store.get_skill("reviewer")
    assert parent.status == expected_status
    assert parent.latest_version == created.version


def test_update_skill_version_without_status_is_a_no_op(store):
    _seed_skill(store, [(1, SkillStatus.ACTIVE)])

    assert store.update_skill_version("reviewer", 1).status == SkillStatus.ACTIVE


def test_update_skill_version_records_last_updated_by(store):
    _seed_skill(store, [(1, SkillStatus.DRAFT)])

    store.update_skill_version("reviewer", 1, status=SkillStatus.ACTIVE, last_updated_by="alice")

    assert _get_version_audit(store, 1)[0] == "alice"


@pytest.mark.parametrize(
    ("current", "status", "message"),
    [
        (SkillStatus.DRAFT, None, "status cannot be null"),
        (SkillStatus.DRAFT, "invalid", "Invalid SkillVersion status"),
        (SkillStatus.DRAFT, SkillStatus.DRAFT, "Invalid status transition"),
        (SkillStatus.DRAFT, SkillStatus.DEPRECATED, "Invalid status transition"),
        (SkillStatus.ACTIVE, SkillStatus.ACTIVE, "Invalid status transition"),
        (SkillStatus.ACTIVE, SkillStatus.DELETED, "Invalid status transition"),
        (SkillStatus.DEPRECATED, SkillStatus.DRAFT, "Invalid status transition"),
        (SkillStatus.DEPRECATED, SkillStatus.DEPRECATED, "Invalid status transition"),
    ],
)
def test_rejected_skill_version_update_preserves_metadata_and_aliases(
    store, current, status, message
):
    store.create_skill_version(
        "reviewer",
        status=SkillStatus.DRAFT if current == SkillStatus.DRAFT else SkillStatus.ACTIVE,
        created_by="alice",
        source_type=SkillSourceType.GIT,
        source="https://example.com/reviewer.git",
    )
    if current == SkillStatus.DEPRECATED:
        store.update_skill_version("reviewer", 1, status=current, last_updated_by="alice")
    store.set_skill_alias("reviewer", "preview", 1)
    store.set_skill_version_tag("reviewer", 1, "team", "platform")
    parent = store.get_skill("reviewer")
    version = store.get_skill_version("reviewer", 1)
    assert parent.source_type == SkillSourceType.GIT

    with pytest.raises(MlflowException, match=message) as exc:
        store.update_skill_version("reviewer", 1, status=status, last_updated_by="bob")

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert store.get_skill("reviewer") == parent
    assert store.get_skill_version("reviewer", 1) == version
    assert store.get_skill_version_by_alias("reviewer", "preview") == version


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (SkillStatus.DRAFT, SkillStatus.ACTIVE),
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


@pytest.mark.parametrize(
    "kwargs",
    [{}, *({"status": status} for status in SkillStatus)],
    ids=["omitted", "draft", "active", "deprecated", "deleted"],
)
def test_update_skill_version_of_deleted_version_is_not_found(store, kwargs):
    store.create_skill_version(
        "reviewer",
        status=SkillStatus.DRAFT,
        created_by="alice",
        source_type=SkillSourceType.GIT,
        source="https://example.com/reviewer.git",
    )
    store.set_skill_version_tag("reviewer", 1, "team", "platform")
    store.set_skill_alias("reviewer", "preview", 1)
    store.delete_skill_version("reviewer", 1, last_updated_by="alice")
    parent = store.get_skill("reviewer")
    with store.ManagedSessionMaker() as session:
        version = store._get_skill_version_or_raise(
            session, "reviewer", 1, "", include_deleted=True
        ).to_mlflow_entity()
    assert version.status == SkillStatus.DELETED
    assert version.tags == {"team": "platform"}
    assert version.aliases == []

    with pytest.raises(
        MlflowException, match="Skill version 'reviewer' version '1' not found"
    ) as exc:
        store.update_skill_version("reviewer", 1, last_updated_by="bob", **kwargs)

    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    assert store.get_skill("reviewer") == parent
    with store.ManagedSessionMaker() as session:
        assert (
            store._get_skill_version_or_raise(
                session, "reviewer", 1, "", include_deleted=True
            ).to_mlflow_entity()
        ) == version
    assert _get_alias_rows(store) == []


def test_update_skill_version_does_not_flush_duplicate_update(store):
    _seed_skill(store, [(1, SkillStatus.DRAFT)])
    statements = []

    def capture_statement(conn, cursor, statement, parameters, context, executemany):
        if (
            statement.lstrip().upper().startswith("UPDATE")
            and "SKILL_VERSIONS" in statement.upper()
        ):
            statements.append(statement)

    sqlalchemy.event.listen(store.engine, "before_cursor_execute", capture_statement)
    try:
        store.update_skill_version("reviewer", 1, status=SkillStatus.ACTIVE)
    finally:
        sqlalchemy.event.remove(store.engine, "before_cursor_execute", capture_statement)

    assert len(statements) == 1


def test_concurrent_update_skill_version_returns_conflict(store):
    _seed_skill(store, [(1, SkillStatus.ACTIVE)])
    barrier = threading.Barrier(2, timeout=30)
    original = store._get_skill_version_or_raise

    def synchronized_read(*args, **kwargs):
        result = original(*args, **kwargs)
        barrier.wait()
        return result

    def update(status):
        try:
            store.update_skill_version("reviewer", 1, status=status)
            return "SUCCESS"
        except MlflowException as e:
            return e.error_code

    with mock.patch.object(
        store, "_get_skill_version_or_raise", side_effect=synchronized_read
    ) as get_version_mock:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="skill-update-race") as executor:
            results = list(executor.map(update, [SkillStatus.DRAFT, SkillStatus.DEPRECATED]))

    assert get_version_mock.call_count == 2
    assert sorted(results) == ["RESOURCE_CONFLICT", "SUCCESS"]
    assert _get_version_row(store, 1) in {
        SkillStatus.DRAFT.value,
        SkillStatus.DEPRECATED.value,
    }


def test_concurrent_delete_skill_version_returns_conflict(store):
    _seed_skill(store, [(1, SkillStatus.DEPRECATED)])
    barrier = threading.Barrier(2, timeout=30)
    original = store._get_skill_version_or_raise

    def synchronized_read(*args, **kwargs):
        result = original(*args, **kwargs)
        barrier.wait()
        return result

    def delete():
        try:
            store.delete_skill_version("reviewer", 1)
            return "SUCCESS"
        except MlflowException as e:
            return e.error_code

    with mock.patch.object(
        store, "_get_skill_version_or_raise", side_effect=synchronized_read
    ) as get_version_mock:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="skill-delete-race") as executor:
            results = list(executor.map(lambda _: delete(), range(2)))

    assert get_version_mock.call_count == 2
    assert sorted(results) == ["RESOURCE_CONFLICT", "SUCCESS"]
    assert _get_version_row(store, 1) == SkillStatus.DELETED.value


def test_concurrent_set_skill_alias_and_delete_leave_no_deleted_alias(store):
    _seed_skill(store, [(1, SkillStatus.DEPRECATED)])
    barrier = threading.Barrier(2, timeout=30)
    thread_state = threading.local()

    def synchronize_first_version_read(conn, cursor, statement, parameters, context, executemany):
        if (
            getattr(thread_state, "wait_for_version_read", False)
            and statement.lstrip().upper().startswith("SELECT")
            and "FROM SKILL_VERSIONS" in statement.upper()
        ):
            thread_state.wait_for_version_read = False
            barrier.wait()

    def set_alias():
        thread_state.wait_for_version_read = True
        try:
            store.set_skill_alias("reviewer", "production", 1)
            return "SUCCESS"
        except MlflowException as e:
            return e.error_code

    def delete():
        thread_state.wait_for_version_read = True
        try:
            store.delete_skill_version("reviewer", 1)
            return "SUCCESS"
        except MlflowException as e:
            return e.error_code

    sqlalchemy.event.listen(store.engine, "before_cursor_execute", synchronize_first_version_read)
    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="skill-alias-race") as executor:
            results = list(executor.map(lambda operation: operation(), [set_alias, delete]))
    finally:
        sqlalchemy.event.remove(
            store.engine, "before_cursor_execute", synchronize_first_version_read
        )

    assert sorted(results) in (["RESOURCE_DOES_NOT_EXIST", "SUCCESS"], ["SUCCESS", "SUCCESS"])
    assert _get_version_row(store, 1) == SkillStatus.DELETED.value
    assert _get_alias_rows(store) == []


def test_set_skill_alias_retries_integrity_error(store):
    _seed_skill(store, [(1, SkillStatus.DRAFT)])
    flushes = 0

    def fail_first_flush(session, flush_context, instances):
        nonlocal flushes
        flushes += 1
        if flushes == 1:
            raise sqlalchemy.exc.IntegrityError("forced alias conflict", None, None)

    sqlalchemy.event.listen(sqlalchemy.orm.Session, "before_flush", fail_first_flush)
    try:
        store.set_skill_alias("reviewer", "production", 1)
    finally:
        sqlalchemy.event.remove(sqlalchemy.orm.Session, "before_flush", fail_first_flush)

    assert flushes == 2
    with store.ManagedSessionMaker() as session:
        aliases = (
            store
            ._get_query(session, SqlSkillAlias)
            .filter(
                SqlSkillAlias.name == "reviewer",
                SqlSkillAlias.organization == "",
            )
            .all()
        )
        assert len(aliases) == 1
        assert aliases[0].alias == "production"
        assert aliases[0].version == 1


@pytest.mark.parametrize("status", [SkillStatus.DRAFT, SkillStatus.DEPRECATED])
@pytest.mark.parametrize("method", ["delete_skill_version", "update_skill_version"])
def test_delete_skill_version_soft_deletes_and_removes_aliases(store, status, method):
    _seed_skill(store, [(1, SkillStatus.DRAFT), (2, status)])
    store.set_skill_alias("reviewer", "preview", 2)
    store.set_skill_alias("reviewer", "production", 2)
    store.set_skill_alias("reviewer", "retained", 1)
    store.set_skill_version_tag("reviewer", 2, "team", "platform")
    retained = store.get_skill_version("reviewer", 1)
    parent = store.get_skill("reviewer")
    assert parent.latest_version == 2

    kwargs = {"status": SkillStatus.DELETED} if method == "update_skill_version" else {}
    result = getattr(store, method)("reviewer", 2, last_updated_by="alice", **kwargs)

    if method == "update_skill_version":
        assert result.status == SkillStatus.DELETED
        assert result.aliases == []
    else:
        assert result is None
    with store.ManagedSessionMaker() as session:
        deleted = store._get_skill_version_or_raise(
            session, "reviewer", 2, "", include_deleted=True
        ).to_mlflow_entity()
    assert deleted.status == SkillStatus.DELETED
    assert deleted.last_updated_by == "alice"
    assert deleted.tags == {"team": "platform"}
    assert deleted.aliases == []
    assert _get_alias_rows(store) == [("retained", 1)]
    for alias in ("preview", "production"):
        with pytest.raises(MlflowException, match="not found") as exc:
            store.get_skill_version_by_alias("reviewer", alias)
        assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    with pytest.raises(MlflowException, match="not found") as exc:
        store.get_skill_version("reviewer", 2)
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    assert store.search_skill_versions("reviewer") == [retained]
    assert store.get_skill_version("reviewer", 1) == retained
    assert store.get_skill_version_by_alias("reviewer", "retained") == retained
    assert store.get_latest_skill_version("reviewer") == retained
    parent.latest_version = 1
    parent.status = SkillStatus.DRAFT
    parent.aliases = {"retained": 1}
    assert store.get_skill("reviewer") == parent


def test_delete_skill_version_rejects_active_versions(store):
    _seed_skill(
        store,
        [(1, SkillStatus.ACTIVE)],
        source_type=SkillSourceType.GIT,
        source="https://example.com/reviewer.git",
    )
    store.set_skill_alias("reviewer", "production", 1)
    parent = store.get_skill("reviewer")
    version = store.get_skill_version("reviewer", 1)
    assert parent.source_type == SkillSourceType.GIT

    with pytest.raises(
        MlflowException, match="Invalid status transition from 'active' to 'deleted'"
    ) as exc:
        store.delete_skill_version("reviewer", 1, last_updated_by="bob")

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert _get_version_row(store, 1) == SkillStatus.ACTIVE.value
    assert store.get_skill("reviewer") == parent
    assert store.get_skill_version("reviewer", 1) == version
    assert store.get_skill_version_by_alias("reviewer", "production") == version


def test_delete_skill_version_of_deleted_version_is_not_found(store):
    _seed_skill(store, [(1, SkillStatus.DEPRECATED)])

    store.delete_skill_version("reviewer", 1, last_updated_by="alice")
    first_audit = _get_version_audit(store, 1)

    with pytest.raises(
        MlflowException, match="Skill version 'reviewer' version '1' not found"
    ) as exc:
        store.delete_skill_version("reviewer", 1, last_updated_by="bob")

    assert _get_version_row(store, 1) == SkillStatus.DELETED.value
    assert _get_version_audit(store, 1) == first_audit
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


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


@pytest.mark.parametrize(
    ("versions", "expected_version", "expected_status"),
    [
        pytest.param([], None, None, id="empty-parent"),
        pytest.param([(1, SkillStatus.DELETED)], None, None, id="all-deleted"),
        pytest.param(
            [(9, SkillStatus.ACTIVE), (10, SkillStatus.ACTIVE)],
            10,
            SkillStatus.ACTIVE,
            id="numeric-active-order",
        ),
        pytest.param(
            [
                (1, SkillStatus.ACTIVE),
                (2, SkillStatus.ACTIVE),
                (3, SkillStatus.DRAFT),
                (4, SkillStatus.DEPRECATED),
            ],
            2,
            SkillStatus.ACTIVE,
            id="active-before-higher-non-active",
        ),
        pytest.param(
            [(9, SkillStatus.DEPRECATED), (10, SkillStatus.DRAFT), (11, SkillStatus.DELETED)],
            10,
            SkillStatus.DRAFT,
            id="draft-fallback",
        ),
        pytest.param(
            [(9, SkillStatus.DRAFT), (10, SkillStatus.DEPRECATED), (11, SkillStatus.DELETED)],
            10,
            SkillStatus.DEPRECATED,
            id="deprecated-fallback",
        ),
    ],
)
def test_latest_skill_resolution_agrees_with_parent_and_alias(
    store, versions, expected_version, expected_status
):
    _seed_skill(
        store,
        versions,
        organization="acme",
        source_type=SkillSourceType.GIT,
        source="https://example.com/reviewer.git",
    )
    parent = store.get_skill("reviewer", organization="acme")
    assert parent.latest_version == expected_version
    assert parent.status == expected_status

    if expected_version is None:
        assert parent.source_type is None
        with pytest.raises(MlflowException, match="No resolved latest version") as exc:
            store.get_latest_skill_version("reviewer", organization="acme")
        assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
        with pytest.raises(MlflowException, match="No resolved latest version") as exc:
            store.get_skill_version_by_alias("reviewer", "latest", organization="acme")
        assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    else:
        assert parent.source_type == SkillSourceType.GIT
        expected = store.get_skill_version("reviewer", expected_version, organization="acme")
        assert parent.source_type == expected.source_type
        assert store.get_latest_skill_version("reviewer", organization="acme") == expected
        assert (
            store.get_skill_version_by_alias("reviewer", "latest", organization="acme") == expected
        )
        assert expected.status == expected_status


def test_latest_skill_version_does_not_return_concurrently_deleted_version(store):
    _seed_skill(store, [(1, SkillStatus.ACTIVE)])
    original_skill_version_query = store._skill_version_query

    def delete_version_before_lookup(session):
        with store.ManagedSessionMaker(read_only=False) as delete_session:
            delete_session.query(SqlSkillVersion).filter(
                SqlSkillVersion.name == "reviewer",
                SqlSkillVersion.organization == "",
                SqlSkillVersion.version == 1,
            ).update({SqlSkillVersion.status: SkillStatus.DELETED.value})
        return original_skill_version_query(session)

    with mock.patch.object(
        store, "_skill_version_query", side_effect=delete_version_before_lookup
    ) as skill_version_query_mock:
        with pytest.raises(MlflowException, match="No resolved latest version") as exc:
            store.get_latest_skill_version("reviewer")

    assert skill_version_query_mock.call_count == 1
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_latest_skill_version_recomputes_after_status_change(store):
    _seed_skill(store, [(1, SkillStatus.ACTIVE), (2, SkillStatus.DRAFT)])
    original_skill_version_query = store._skill_version_query

    def deprecate_version_before_lookup(session):
        with store.ManagedSessionMaker(read_only=False) as update_session:
            update_session.query(SqlSkillVersion).filter(
                SqlSkillVersion.name == "reviewer",
                SqlSkillVersion.organization == "",
                SqlSkillVersion.version == 1,
            ).update({SqlSkillVersion.status: SkillStatus.DEPRECATED.value})
        return original_skill_version_query(session)

    with mock.patch.object(
        store, "_skill_version_query", side_effect=deprecate_version_before_lookup
    ) as skill_version_query_mock:
        assert store.get_latest_skill_version("reviewer").version == 2

    assert skill_version_query_mock.call_count == 1


def test_latest_alias_is_dynamic_and_cannot_be_stored(store):
    _seed_skill(store, [(1, SkillStatus.ACTIVE), (2, SkillStatus.ACTIVE)])

    with pytest.raises(MlflowException, match="reserved") as exc:
        store.set_skill_alias("reviewer", "latest", 1)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert store.get_skill_version_by_alias("reviewer", "latest").version == 2


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
    store.set_skill_alias("reviewer", "stable", 1)

    # Both a new alias and a retarget of an existing one are refused.
    for alias in ("fresh", "stable"):
        with pytest.raises(MlflowException, match="not found") as exc:
            store.set_skill_alias("reviewer", alias, 99)
        assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"

        with pytest.raises(MlflowException, match="deleted") as exc:
            store.set_skill_alias("reviewer", alias, 2)
        assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"

    # No new alias was created, and the rejected retarget left "stable" on its old version.
    assert _get_alias_rows(store) == [("stable", 1)]

    _add_alias(store, "stale", 2)
    with pytest.raises(MlflowException, match="missing or deleted"):
        store.get_skill_version_by_alias("reviewer", "stale")


def test_aliases_are_scoped_to_their_organization(store):
    # Every organization has its own `reviewer` v1, so an alias that ignored the organization
    # would still find a version.
    for organization in ("", "acme", "globex"):
        _seed_skill(store, [(1, SkillStatus.ACTIVE)], organization=organization)
    store.set_skill_alias("reviewer", "stable", 1, organization="acme")

    for organization in ("", "globex"):
        with pytest.raises(MlflowException, match="Alias 'stable'") as exc:
            store.get_skill_version_by_alias("reviewer", "stable", organization=organization)
        assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
        with pytest.raises(MlflowException, match="Alias 'stable'"):
            store.delete_skill_alias("reviewer", "stable", organization=organization)

    found = store.get_skill_version_by_alias("reviewer", "stable", organization="acme")
    assert (found.organization, found.version) == ("acme", 1)


def test_deleting_a_missing_alias_raises(store):
    _seed_skill(store, [(1, SkillStatus.ACTIVE)])

    with pytest.raises(MlflowException, match="Alias 'missing'") as exc:
        store.delete_skill_alias("reviewer", "missing")

    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_delete_skill_alias_preserves_version_and_other_aliases(store):
    store.create_skill_version(
        "reviewer",
        organization="acme",
        status=SkillStatus.DRAFT,
        source_type=SkillSourceType.GIT,
        source="https://example.com/reviewer.git",
    )
    store.set_skill_alias("reviewer", "preview", 1, organization="acme")
    store.set_skill_alias("reviewer", "stable", 1, organization="acme")
    store.set_skill_version_tag("reviewer", 1, "team", "platform", organization="acme")
    parent = store.get_skill("reviewer", organization="acme")
    version = store.get_skill_version("reviewer", 1, organization="acme")
    assert parent.source_type == SkillSourceType.GIT

    assert store.delete_skill_alias("reviewer", "preview", organization="acme") is None

    with pytest.raises(MlflowException, match="Alias 'preview' not found") as exc:
        store.get_skill_version_by_alias("reviewer", "preview", organization="acme")
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    parent.aliases = {"stable": 1}
    version.aliases = ["stable"]
    assert store.get_skill("reviewer", organization="acme") == parent
    assert store.get_skill_version("reviewer", 1, organization="acme") == version
    assert store.get_skill_version_by_alias("reviewer", "stable", organization="acme") == version


def test_deleting_latest_alias_is_invalid(store):
    _seed_skill(store, [(1, SkillStatus.ACTIVE)])

    with pytest.raises(MlflowException, match="cannot be deleted") as exc:
        store.delete_skill_alias("reviewer", "latest")

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_skill_alias_validation_uses_skill_specific_error(store):
    with pytest.raises(MlflowException, match="Skill alias name cannot be empty"):
        store.set_skill_alias("reviewer", "", 1)
