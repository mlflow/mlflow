from datetime import date
from unittest import mock

import pytest

from mlflow.entities import Feedback
from mlflow.entities.assessment import AssessmentSource
from mlflow.entities.entity_type import EntityAssociationType
from mlflow.entities.workspace import Workspace, WorkspaceDeletionMode
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import (
    SqlEntityAssociation,
    SqlIssue,
    SqlTraceInfo,
    SqlTraceMetricDailyRollup,
)
from mlflow.utils.workspace_context import WorkspaceContext

from tests.store.tracking.sqlalchemy_store.test_trace_backend_associations import (
    MemoryTraceBackend,
    create_trace,
)


@pytest.fixture
def backend_store(store):
    backend = MemoryTraceBackend()
    host = type(store)(store.db_uri, store.artifact_root_uri, trace_backend=backend)
    try:
        yield host, backend
    finally:
        host._dispose_engine()


def test_host_orders_run_hooks_after_native_commit(backend_store):
    store, backend = backend_store
    exp_id = store.create_experiment("hooks")
    run = store.create_run(exp_id, "user", 1, [], "run")
    create_trace(store, exp_id, "tr-a")
    store.create_assessment(
        Feedback(
            name="quality",
            value=True,
            trace_id="tr-a",
            source=AssessmentSource(source_type="HUMAN", source_id="user"),
            metadata={"mlflow.assessment.sourceRunId": run.info.run_id},
        )
    )
    issue = store.create_issue(exp_id, "Issue", "", source_run_id=run.info.run_id)
    store.link_traces_to_run(["tr-a"], run.info.run_id)
    with mock.patch.object(backend, "on_run_deleted", wraps=backend.on_run_deleted) as deleted:
        store.delete_run(run.info.run_id)
    deleted.assert_called_once_with(run.info.run_id)
    assert backend.assessments == []
    with mock.patch.object(
        backend, "on_run_permanently_deleted", wraps=backend.on_run_permanently_deleted
    ) as permanent:
        store._hard_delete_run(run.info.run_id)
    permanent.assert_called_once_with(run.info.run_id)
    assert store.get_issue(issue.issue_id).source_run_id is None
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlEntityAssociation).count() == 0


def test_soft_experiment_deletion_preserves_traces_and_issues(backend_store):
    store, backend = backend_store
    exp_id = store.create_experiment("experiment-hooks")
    run = store.create_run(exp_id, "user", 1, [], "run")
    create_trace(store, exp_id, "tr-a")
    issue = store.create_issue(exp_id, "Issue", "")
    store.create_assessment(
        Feedback(
            name="quality",
            value=True,
            trace_id="tr-a",
            source=AssessmentSource(source_type="HUMAN", source_id="user"),
            metadata={"mlflow.assessment.sourceRunId": run.info.run_id},
        )
    )
    store.delete_experiment(exp_id)
    assert backend.assessments == []
    assert store.get_trace_info("tr-a").trace_id == "tr-a"
    assert store.get_issue(issue.issue_id).name == "Issue"
    store._hard_delete_run(run.info.run_id)
    store._hard_delete_experiment(exp_id)
    assert backend.traces == {}
    assert backend.issues == {}
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlIssue).count() == 0


def test_backend_issue_only_experiment_cleanup_does_not_depend_on_traces(backend_store):
    store, backend = backend_store
    exp_id = store.create_experiment("issue-only")
    issue = store.create_issue(exp_id, "Issue without assessments", "")
    assert backend.traces == {}
    assert store.search_issues(experiment_id=exp_id, include_trace_count=True)[0].trace_count == 0
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlIssue).count() == 0
    store.delete_experiment(exp_id)
    assert store.get_issue(issue.issue_id).name == issue.name
    store._hard_delete_experiment(exp_id)
    assert backend.issues == {}
    backend.on_experiment_permanently_deleted(exp_id)
    assert backend.issues == {}


def test_backend_issue_only_workspace_is_restricted_and_cleans_only_named_scope(backend_store):
    store, backend = backend_store
    with WorkspaceContext("issue-team-a"):
        exp_a = store.create_experiment("A issue-only")
        store.create_issue(exp_a, "Issue A", "")
    with WorkspaceContext("issue-team-b"):
        exp_b = store.create_experiment("B issue-only")
        issue_b = store.create_issue(exp_b, "Issue B", "")
        with pytest.raises(MlflowException, match="resources remain"):
            store.before_workspace_deleted("issue-team-a", WorkspaceDeletionMode.RESTRICT)
        store.on_workspace_deleted("issue-team-a")
        store.on_workspace_deleted("issue-team-a")
        assert store.get_issue(issue_b.issue_id).name == "Issue B"
    assert {workspace for workspace, _ in backend.issues.values()} == {"issue-team-b"}


def test_workspace_cascade_cleans_backend_issues_after_sql_parents_are_gone(backend_store):
    store, backend = backend_store
    if not store.supports_workspaces:
        pytest.skip("Workspace provider cascade requires workspace mode")
    provider = store._get_workspace_provider_instance()
    for workspace in ("issue-cascade-a", "issue-cascade-b"):
        provider.create_workspace(Workspace(name=workspace))
        with WorkspaceContext(workspace):
            exp_id = store.create_experiment(f"exp-{workspace}")
            store.create_issue(exp_id, "Issue without traces", "")
    assert backend.traces == {}
    with WorkspaceContext("issue-cascade-b"):
        store.before_workspace_deleted("issue-cascade-a", WorkspaceDeletionMode.CASCADE)
        provider.delete_workspace("issue-cascade-a", mode=WorkspaceDeletionMode.CASCADE)
        store.on_workspace_deleted("issue-cascade-a")
        store.on_workspace_deleted("issue-cascade-a")
        assert len(store.search_issues()) == 1
    assert {workspace for workspace, _ in backend.issues.values()} == {"issue-cascade-b"}


def test_hook_failures_are_logged_without_undoing_primary_deletion(backend_store, caplog):
    store, backend = backend_store
    exp_id = store.create_experiment("failed-cleanup")
    run = store.create_run(exp_id, "user", 1, [], "run")
    with mock.patch.object(
        backend, "on_run_deleted", side_effect=RuntimeError("orphan risk")
    ) as hook:
        store.delete_run(run.info.run_id)
    hook.assert_called_once()
    assert store.get_run(run.info.run_id).info.lifecycle_stage == "deleted"
    assert "orphan risk" in caplog.text


def test_failed_parent_operation_does_not_invoke_hook(backend_store):
    store, backend = backend_store
    with mock.patch.object(backend, "on_run_deleted") as hook:
        with pytest.raises(MlflowException, match="not found"):
            store.delete_run("missing")
    hook.assert_not_called()


def test_workspace_reassignment_rejected_before_backend_mutation(backend_store):
    store, backend = backend_store
    with mock.patch.object(backend, "before_workspace_deleted") as before:
        with pytest.raises(MlflowException, match="reassignment is not supported"):
            store.before_workspace_deleted("team", WorkspaceDeletionMode.SET_DEFAULT)
    before.assert_not_called()


def test_native_sql_trace_deletion_physically_removes_associations(store):
    exp_id = store.create_experiment("native-cleanup")
    run = store.create_run(exp_id, "user", 1, [], "run")
    create_trace(store, exp_id, "tr-sql")
    store.link_traces_to_run(["tr-sql"], run.info.run_id)
    assert store.delete_traces(exp_id, trace_ids=["tr-sql"]) == 1
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlEntityAssociation).count() == 0
        assert session.query(SqlTraceInfo).count() == 0


def test_native_sql_unlink_is_idempotent_and_preserves_other_links(store):
    exp_id = store.create_experiment("native-unlink")
    runs = [store.create_run(exp_id, "user", 1, [], name) for name in ("a", "b")]
    create_trace(store, exp_id, "tr-unlink")
    for run in runs:
        store.link_traces_to_run(["tr-unlink"], run.info.run_id)
    store.unlink_traces_from_run(["tr-unlink"], runs[0].info.run_id)
    store.unlink_traces_from_run(["tr-unlink"], runs[0].info.run_id)
    assert store.get_trace_info("tr-unlink").trace_id == "tr-unlink"
    with store.ManagedSessionMaker() as session:
        assert [row.destination_id for row in session.query(SqlEntityAssociation).all()] == [
            runs[1].info.run_id
        ]


def test_native_sql_unlink_preserves_workspace_isolation(store):
    if not store.supports_workspaces:
        pytest.skip("Workspace isolation requires workspace mode")
    with WorkspaceContext("unlink-a"):
        exp_a = store.create_experiment("A")
        run_a = store.create_run(exp_a, "user", 1, [], "a")
        create_trace(store, exp_a, "tr-unlink-a")
        store.link_traces_to_run(["tr-unlink-a"], run_a.info.run_id)
    with WorkspaceContext("unlink-b"):
        exp_b = store.create_experiment("B")
        run_b = store.create_run(exp_b, "user", 1, [], "b")
        with pytest.raises(MlflowException, match="not found"):
            store.unlink_traces_from_run(["tr-unlink-a"], run_a.info.run_id)
        store.unlink_traces_from_run(["tr-unlink-a"], run_b.info.run_id)
    with WorkspaceContext("unlink-a"):
        assert list(
            store.search_entities_by_source(
                "tr-unlink-a", EntityAssociationType.TRACE, EntityAssociationType.RUN
            )
        ) == [run_a.info.run_id]


def test_native_hard_experiment_deletion_cleans_orphanable_trace_links(store):
    exp_id = store.create_experiment("native-experiment-cleanup")
    run = store.create_run(exp_id, "user", 1, [], "run")
    create_trace(store, exp_id, "tr-sql")
    store.link_traces_to_run(["tr-sql"], run.info.run_id)
    store.delete_experiment(exp_id)
    store._hard_delete_experiment(exp_id)
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlEntityAssociation).count() == 0
        assert session.query(SqlTraceInfo).count() == 0


def test_native_workspace_cascade_removes_rollups_and_trace_associations(store):
    if not store.supports_workspaces:
        pytest.skip("Workspace provider cascade requires workspace mode")
    provider = store._get_workspace_provider_instance()
    provider.create_workspace(Workspace(name="native-cleanup"))
    with WorkspaceContext("native-cleanup"):
        exp_id = store.create_experiment("native-workspace")
        run = store.create_run(exp_id, "user", 1, [], "run")
        create_trace(store, exp_id, "tr-native-workspace")
        store.link_traces_to_run(["tr-native-workspace"], run.info.run_id)
        store.create_issue(exp_id, "Issue", "")
        with store.ManagedSessionMaker(read_only=False) as session:
            session.add(
                SqlTraceMetricDailyRollup(
                    experiment_id=int(exp_id),
                    rollup_day=date(2020, 1, 1),
                    metric_name="trace_count",
                    grouping_set="global",
                    sample_count=1,
                )
            )
    provider.delete_workspace("native-cleanup", mode=WorkspaceDeletionMode.CASCADE)
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlTraceMetricDailyRollup).count() == 0
        assert session.query(SqlEntityAssociation).count() == 0
        assert session.query(SqlIssue).count() == 0
        assert session.query(SqlTraceInfo).count() == 0


from datetime import date
