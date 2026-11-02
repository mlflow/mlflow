import asyncio
import inspect
from dataclasses import replace
from unittest import mock

import pytest

from mlflow.entities import Issue, IssueReference, IssueSeverity, IssueStatus, TraceInfo, ViewType
from mlflow.entities.entity_type import EntityAssociationType
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.entities.workspace import WorkspaceDeletionMode
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.abstract_trace_backend import AbstractTraceBackend
from mlflow.store.tracking.dbmodels.models import (
    SqlAssessments,
    SqlEntityAssociation,
    SqlIssue,
    SqlTraceInfo,
)
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.tracing.constant import AssessmentMetadataKey, SpansLocation, TraceTagKey
from mlflow.utils.validation import validate_delete_traces_parameters
from mlflow.utils.workspace_context import WorkspaceContext, get_request_workspace


class MemoryTraceBackend(AbstractTraceBackend):
    """Test storage plugin owning issues and assessments but not SQL associations."""

    def __init__(self):
        self.traces = {}
        self.assessments = []
        self.issues = {}

    def start_trace(self, trace_info):
        trace_info.tags[TraceTagKey.SPANS_LOCATION] = SpansLocation.TRACKING_STORE.value
        self.traces[trace_info.trace_id] = (get_request_workspace(), trace_info)
        return trace_info

    def get_trace_info(self, trace_id):
        if trace_id not in self.traces or self.traces[trace_id][0] != get_request_workspace():
            raise MlflowException("Trace not found", RESOURCE_DOES_NOT_EXIST)
        info = self.traces[trace_id][1]
        self._tracking_store.get_experiment(info.experiment_id)
        return info

    def batch_get_trace_infos(self, trace_ids, location=None, experiment_ids=None):
        found = []
        for trace_id in trace_ids:
            try:
                info = self.get_trace_info(trace_id)
            except MlflowException as e:
                if e.error_code != "RESOURCE_DOES_NOT_EXIST":
                    raise
            else:
                if experiment_ids is None or info.experiment_id in experiment_ids:
                    found.append(info)
        return found

    def log_spans(self, location, spans, tracking_uri=None):
        return spans

    def delete_traces(
        self, experiment_id, max_timestamp_millis=None, max_traces=None, trace_ids=None
    ):
        validate_delete_traces_parameters(max_timestamp_millis, max_traces, trace_ids)
        candidates = [
            info
            for workspace, info in self.traces.values()
            if workspace == get_request_workspace()
            and info.experiment_id == experiment_id
            and (trace_ids is None or info.trace_id in trace_ids)
            and (max_timestamp_millis is None or info.request_time <= max_timestamp_millis)
        ]
        ids = [
            info.trace_id
            for info in sorted(candidates, key=lambda info: (info.request_time, info.trace_id))[
                :max_traces
            ]
        ]
        self._delete_ids(ids)
        return len(ids)

    def _delete_ids(self, trace_ids):
        self.assessments = [a for a in self.assessments if a.trace_id not in trace_ids]
        for trace_id in trace_ids:
            del self.traces[trace_id]
        super().on_traces_deleted(trace_ids)

    def create_assessment(self, assessment):
        self.get_trace_info(assessment.trace_id)
        self.assessments.append(assessment)
        return assessment

    def create_issue(self, experiment_id, name, description, **kwargs):
        self._tracking_store.get_experiment(experiment_id)
        if source_run_id := kwargs.get("source_run_id"):
            self._tracking_store.get_run(source_run_id)
        issue_id = kwargs.pop("issue_id", None) or f"iss-memory-{len(self.issues)}"
        issue = Issue(
            issue_id=issue_id,
            experiment_id=str(experiment_id),
            name=name,
            description=description,
            created_timestamp=len(self.issues),
            last_updated_timestamp=len(self.issues),
            status=kwargs.pop("status", IssueStatus.PENDING),
            **kwargs,
        )
        self.issues[issue_id] = (get_request_workspace(), issue)
        return issue

    def get_issue(self, issue_id):
        if issue_id not in self.issues or self.issues[issue_id][0] != get_request_workspace():
            raise MlflowException("Issue not found", RESOURCE_DOES_NOT_EXIST)
        issue = self.issues[issue_id][1]
        self._tracking_store.get_experiment(issue.experiment_id)
        return issue

    def update_issue(self, issue_id, **kwargs):
        issue = replace(
            self.get_issue(issue_id),
            **{key: value for key, value in kwargs.items() if value is not None},
        )
        self.issues[issue_id] = (get_request_workspace(), issue)
        return issue

    def search_issues(
        self,
        experiment_id=None,
        filter_string=None,
        max_results=None,
        page_token=None,
        include_trace_count=False,
        *,
        name=None,
        statuses=None,
        severities=None,
        categories=None,
        order_by=None,
        include_total_count=False,
        view=None,
    ):
        issues = [
            replace(
                issue,
                trace_count=len({
                    a.trace_id
                    for a in self.assessments
                    if isinstance(a, IssueReference)
                    and a.name == issue.issue_id
                    and self.traces[a.trace_id][0] == get_request_workspace()
                })
                if include_trace_count
                else None,
            )
            for workspace, issue in self.issues.values()
            if workspace == get_request_workspace()
            and (experiment_id is None or issue.experiment_id == str(experiment_id))
            and (name is None or name.lower() in issue.name.lower())
            and (not statuses or issue.status in statuses)
            and (not severities or issue.severity in severities)
            and (not categories or set(categories) & set(issue.categories or []))
        ]
        if order_by:
            issues.sort(key=lambda issue: issue.name.lower())
        else:
            issues.sort(
                key=lambda issue: (
                    issue.severity._rank if issue.severity else -1,
                    issue.trace_count or 0,
                    issue.created_timestamp,
                    issue.issue_id,
                ),
                reverse=True,
            )
        offset = int(page_token.removeprefix("memory-issues:")) if page_token else 0
        end = offset + max_results if max_results is not None else len(issues)
        selected = issues[offset:end]
        if view == "basic":
            selected = [
                replace(issue, description="", metadata=None, root_causes=None)
                for issue in selected
            ]
        return PagedList(
            selected,
            f"memory-issues:{end}" if end < len(issues) else None,
            len(issues) if include_total_count else None,
        )

    def on_run_deleted(self, run_id):
        self.assessments = [
            a
            for a in self.assessments
            if (a.metadata or {}).get(AssessmentMetadataKey.SOURCE_RUN_ID) != run_id
            or self.traces[a.trace_id][0] != get_request_workspace()
        ]

    def on_run_permanently_deleted(self, run_id):
        self.on_run_deleted(run_id)
        self.issues = {
            issue_id: (workspace, replace(issue, source_run_id=None))
            if workspace == get_request_workspace() and issue.source_run_id == run_id
            else (workspace, issue)
            for issue_id, (workspace, issue) in self.issues.items()
        }

    def on_experiment_deleted(self, experiment_id):
        token = None
        while True:
            runs = self._tracking_store.search_runs(
                [experiment_id], "", ViewType.ALL, max_results=100, page_token=token
            )
            for run in runs:
                self.on_run_deleted(run.info.run_id)
            token = runs.token
            if not token:
                break

    def on_experiment_permanently_deleted(self, experiment_id):
        self._delete_ids([
            trace_id
            for trace_id, (_, info) in self.traces.items()
            if info.experiment_id == experiment_id
        ])
        self.issues = {
            issue_id: (workspace, issue)
            for issue_id, (workspace, issue) in self.issues.items()
            if issue.experiment_id != str(experiment_id)
        }

    def before_workspace_deleted(self, workspace_name, mode):
        if mode == WorkspaceDeletionMode.RESTRICT and (
            any(workspace == workspace_name for workspace, _ in self.traces.values())
            or any(workspace == workspace_name for workspace, _ in self.issues.values())
        ):
            raise MlflowException.invalid_parameter_value("Trace backend resources remain")

    def on_workspace_deleted(self, workspace_name):
        self._delete_ids([
            trace_id
            for trace_id, (workspace, _) in self.traces.items()
            if workspace == workspace_name
        ])
        self.issues = {
            issue_id: (workspace, issue)
            for issue_id, (workspace, issue) in self.issues.items()
            if workspace != workspace_name
        }


@pytest.fixture
def backend_store(db_uri, tmp_path, workspaces_enabled):
    backend = MemoryTraceBackend()
    cls = WorkspaceAwareSqlAlchemyStore if workspaces_enabled else SqlAlchemyStore
    store = cls(db_uri, tmp_path.as_uri(), trace_backend=backend)
    try:
        yield store, backend
    finally:
        store._dispose_engine()


def create_trace(store, experiment_id, trace_id):
    return store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation.from_experiment_id(experiment_id),
            request_time=1,
            execution_duration=1,
            state=TraceState.OK,
        )
    )


def test_backend_owns_issue_queries_while_host_keeps_associations(backend_store):
    store, backend = backend_store
    assert backend._tracking_store is store
    exp_id = store.create_experiment("host-owned")
    run = store.create_run(exp_id, "user", 1, [], "run")
    for trace_id in ("tr-a", "tr-b", "tr-c"):
        create_trace(store, exp_id, trace_id)
        store.link_traces_to_run([trace_id], run.info.run_id)
    first = store.create_issue(exp_id, "A: frequent", "", severity=IssueSeverity.HIGH)
    second = store.create_issue(exp_id, "B: less frequent", "", severity=IssueSeverity.HIGH)
    for trace_id, issue in [("tr-a", first), ("tr-b", first), ("tr-c", second), ("tr-a", first)]:
        store.create_assessment(
            IssueReference(
                issue_id=issue.issue_id,
                issue_name=issue.name,
                trace_id=trace_id,
            )
        )
    page = store.search_issues(
        experiment_id=exp_id, include_trace_count=True, max_results=1, include_total_count=True
    )
    assert [(issue.issue_id, issue.trace_count) for issue in page] == [(first.issue_id, 2)]
    assert page.total_count == 2
    assert page.token == "memory-issues:1"
    page = store.search_issues(
        experiment_id=exp_id,
        include_trace_count=True,
        max_results=1,
        page_token=page.token,
        include_total_count=True,
    )
    assert [(issue.issue_id, issue.trace_count) for issue in page] == [(second.issue_id, 1)]
    assert page.token is None
    explicit = store.search_issues(
        experiment_id=exp_id, include_trace_count=True, order_by=["name ASC"], max_results=1
    )
    assert explicit[0].issue_id == first.issue_id
    assert explicit[0].trace_count == 2
    assert (
        store.update_issue(first.issue_id, status=IssueStatus.RESOLVED).status
        == IssueStatus.RESOLVED
    )
    assert store.get_issue(first.issue_id).name == first.name
    page = store.search_entities_by_destination(
        run.info.run_id, EntityAssociationType.RUN, EntityAssociationType.TRACE, max_results=1
    )
    found = list(page)
    while page.token:
        page = store.search_entities_by_destination(
            run.info.run_id,
            EntityAssociationType.RUN,
            EntityAssociationType.TRACE,
            max_results=1,
            page_token=page.token,
        )
        found.extend(page)
    assert found == ["tr-a", "tr-b", "tr-c"]
    store.link_prompts_to_trace("tr-a", [PromptVersion("prompt", 1, "hello")])
    assert list(
        store.search_entities_by_destination(
            "prompt/1", EntityAssociationType.PROMPT_VERSION, EntityAssociationType.TRACE
        )
    ) == ["tr-a"]
    queue = store.create_review_queue(exp_id, name="Review", queue_type="custom")
    store.add_items_to_review_queue(queue.queue_id, item_ids=["tr-a", "tr-b"])
    assert store.delete_traces(exp_id, trace_ids=["tr-a"]) == 1
    assert [item.item_id for item in store.list_review_queue_items(queue.queue_id)] == ["tr-b"]
    assert all(a.trace_id != "tr-a" for a in backend.assessments)
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlTraceInfo).count() == 0
        assert session.query(SqlAssessments).count() == 0
        assert session.query(SqlIssue).count() == 0
        assert (
            session
            .query(SqlEntityAssociation)
            .filter(SqlEntityAssociation.source_id == "tr-a")
            .count()
            == 0
        )


def test_context_bypass_is_nested_exception_safe_and_store_local(backend_store, db_uri, tmp_path):
    store, backend = backend_store
    other_backend = MemoryTraceBackend()
    other = type(store)(db_uri, tmp_path.as_uri(), trace_backend=other_backend)

    def fail_inside_bypass():
        with store.disable_trace_backend():
            assert store.trace_backend is None
            assert other.trace_backend is other_backend
            with store.disable_trace_backend():
                assert store.trace_backend is None
            assert store.trace_backend is None
            raise RuntimeError("restore")

    try:
        assert store.trace_backend is backend
        with pytest.raises(RuntimeError, match="restore"):
            fail_inside_bypass()
        assert store.trace_backend is backend
    finally:
        other._dispose_engine()


@pytest.mark.asyncio
async def test_context_bypass_is_task_local_and_propagates_to_threads(backend_store):
    store, backend = backend_store
    disabled = asyncio.Event()
    release = asyncio.Event()

    async def native_task():
        with store.disable_trace_backend():
            disabled.set()
            await release.wait()
            assert store.trace_backend is None
            assert await asyncio.to_thread(lambda: store.trace_backend) is None

    task = asyncio.create_task(native_task())
    try:
        await asyncio.wait_for(disabled.wait(), timeout=5)
        assert store.trace_backend is backend
    finally:
        release.set()
        await task
    assert store.trace_backend is backend


def test_hybrid_backend_can_call_native_methods_without_recursion(backend_store):
    store, backend = backend_store
    exp_id = store.create_experiment("hybrid")

    def native_start(info):
        with store.disable_trace_backend():
            return store.start_trace(info)

    with mock.patch.object(backend, "start_trace", side_effect=native_start) as start:
        info = create_trace(store, exp_id, "tr-native")
    start.assert_called_once()
    with store.disable_trace_backend():
        assert store.get_trace_info(info.trace_id).trace_id == "tr-native"
    assert store.trace_backend is backend


def test_reverse_association_search_batches_visibility_and_pages_over_hidden_ids(backend_store):
    store, backend = backend_store
    exp_id = store.create_experiment("batch-associations")
    run = store.create_run(exp_id, "user", 1, [], "run")
    ids = [f"tr-hidden-{i:04d}" for i in range(501)] + ["tr-visible-a", "tr-visible-b"]
    for index, trace_id in enumerate(ids):
        create_trace(store, exp_id, trace_id)
        if index < 501:
            backend.traces[trace_id] = ("other-team", backend.traces[trace_id][1])
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add_all(
            SqlEntityAssociation(
                association_id=f"link-{index}",
                source_id=trace_id,
                source_type=EntityAssociationType.TRACE,
                destination_id=run.info.run_id,
                destination_type=EntityAssociationType.RUN,
                created_time=index,
            )
            for index, trace_id in enumerate(ids)
        )
    with mock.patch.object(
        backend, "batch_get_trace_infos", wraps=backend.batch_get_trace_infos
    ) as batch:
        first = store.search_entities_by_destination(
            run.info.run_id, EntityAssociationType.RUN, EntityAssociationType.TRACE, max_results=1
        )
    assert list(first) == ["tr-visible-a"]
    assert batch.call_count == 2
    assert all(len(call.args[0]) <= 500 for call in batch.call_args_list)
    second = store.search_entities_by_destination(
        run.info.run_id,
        EntityAssociationType.RUN,
        EntityAssociationType.TRACE,
        max_results=1,
        page_token=first.token,
    )
    assert list(second) == ["tr-visible-b"]
    assert second.token is None


def test_host_enforces_workspace_isolation_with_backend_only_traces(backend_store):
    store, backend = backend_store
    if not store.supports_workspaces:
        pytest.skip("Cross-workspace lookup requires workspace mode")
    with WorkspaceContext("backend-a"):
        exp_a = store.create_experiment("A")
        run_a = store.create_run(exp_a, "user", 1, [], "a")
        create_trace(store, exp_a, "tr-a")
        store.link_traces_to_run(["tr-a"], run_a.info.run_id)
        issue_a = store.create_issue(exp_a, "Issue A", "")
    with WorkspaceContext("backend-b"):
        exp_b = store.create_experiment("B")
        run_b = store.create_run(exp_b, "user", 1, [], "b")
        create_trace(store, exp_b, "tr-b")
        store.link_traces_to_run(["tr-b"], run_b.info.run_id)
        store.link_traces_to_run(["tr-a"], run_b.info.run_id)
        with pytest.raises(MlflowException, match="not found"):
            store.link_traces_to_run(["tr-b"], run_a.info.run_id)
        with pytest.raises(MlflowException, match="not found"):
            store.get_issue(issue_a.issue_id)
        assert (
            list(
                store.search_entities_by_source(
                    "tr-a", EntityAssociationType.TRACE, EntityAssociationType.RUN
                )
            )
            == []
        )
        assert (
            list(
                store.search_entities_by_destination(
                    run_a.info.run_id, EntityAssociationType.RUN, EntityAssociationType.TRACE
                )
            )
            == []
        )
        assert list(
            store.search_entities_by_destination(
                run_b.info.run_id, EntityAssociationType.RUN, EntityAssociationType.TRACE
            )
        ) == ["tr-b"]
        assert list(store.search_issues(include_trace_count=True)) == []
    assert len(backend.traces) == 2


@pytest.mark.parametrize(
    "operation",
    [
        name
        for name, value in vars(AbstractTraceBackend).items()
        if inspect.isfunction(value)
        and not name.startswith(("_", "on_", "before_"))
        and name not in ("set_tracking_store", "log_spans_async")
    ],
)
def test_host_routes_complete_trace_contract_with_all_arguments(store, operation):
    backend = mock.create_autospec(AbstractTraceBackend, instance=True)
    host = type(store)(store.db_uri, store.artifact_root_uri, trace_backend=backend)
    try:
        backend.set_tracking_store.assert_called_once_with(host)
        method = getattr(backend, operation)
        arguments = {
            name: getattr(mock.sentinel, name) for name in inspect.signature(method).parameters
        }
        assert getattr(host, operation)(**arguments) is method.return_value
        method.assert_called_once_with(**arguments)
    finally:
        host._dispose_engine()


@pytest.mark.parametrize(
    "destination_type", [EntityAssociationType.RUN, EntityAssociationType.PROMPT_VERSION]
)
def test_reverse_association_pagination_deduplicates_across_destinations(
    backend_store, destination_type
):
    store, _ = backend_store
    exp_id = store.create_experiment("shared-trace-links")
    for trace_id in ("tr-a", "tr-b"):
        create_trace(store, exp_id, trace_id)
    if destination_type == EntityAssociationType.RUN:
        runs = [store.create_run(exp_id, "user", 1, [], name) for name in ("first", "second")]
        destination_ids = [run.info.run_id for run in runs]
        for run in runs:
            store.link_traces_to_run(["tr-a"], run.info.run_id)
        store.link_traces_to_run(["tr-b"], runs[0].info.run_id)
    else:
        prompts = [PromptVersion("prompt", version, "hello") for version in (1, 2)]
        destination_ids = ["prompt/1", "prompt/2"]
        store.link_prompts_to_trace("tr-a", prompts)
        store.link_prompts_to_trace("tr-b", prompts[:1])

    unpaged = store.search_entities_by_destination(
        destination_ids, destination_type, EntityAssociationType.TRACE
    )
    first = store.search_entities_by_destination(
        destination_ids, destination_type, EntityAssociationType.TRACE, max_results=1
    )
    assert list(first) == ["tr-a"]
    assert first.token is not None
    second = store.search_entities_by_destination(
        destination_ids,
        destination_type,
        EntityAssociationType.TRACE,
        max_results=1,
        page_token=first.token,
    )
    assert list(second) == ["tr-b"]
    assert second.token is None
    assert list(first) + list(second) == list(unpaged)


def test_host_associations_work_with_default_batch_metadata_implementation(store):
    class PointReadBackend(MemoryTraceBackend):
        batch_get_trace_infos = AbstractTraceBackend.batch_get_trace_infos

    backend = PointReadBackend()
    host = type(store)(store.db_uri, store.artifact_root_uri, trace_backend=backend)
    try:
        exp_id = host.create_experiment("point-read-backend")
        run = host.create_run(exp_id, "user", 1, [], "run")
        create_trace(host, exp_id, "tr-point")
        host.link_traces_to_run(["tr-point", "missing"], run.info.run_id)
        assert list(
            host.search_entities_by_destination(
                run.info.run_id, EntityAssociationType.RUN, EntityAssociationType.TRACE
            )
        ) == ["tr-point"]
        assert (
            host.batch_get_trace_infos(["tr-point"], experiment_ids=["not-this-experiment"]) == []
        )
    finally:
        host._dispose_engine()


def test_hybrid_backend_can_delegate_issue_operations_to_native_sql(backend_store):
    store, backend = backend_store
    exp_id = store.create_experiment("hybrid-issues")

    def native_create(*args, **kwargs):
        with store.disable_trace_backend():
            return store.create_issue(*args, **kwargs)

    def native_search(*args, **kwargs):
        with store.disable_trace_backend():
            return store.search_issues(*args, **kwargs)

    with mock.patch.object(backend, "create_issue", side_effect=native_create) as create:
        issue = store.create_issue(exp_id, "Native issue", "Stored in SQL")
    create.assert_called_once_with(exp_id, "Native issue", "Stored in SQL")
    with mock.patch.object(backend, "search_issues", side_effect=native_search) as search:
        found = store.search_issues(experiment_id=exp_id, include_trace_count=True)
    search.assert_called_once_with(experiment_id=exp_id, include_trace_count=True)
    assert [(row.issue_id, row.trace_count) for row in found] == [(issue.issue_id, 0)]
    with store.ManagedSessionMaker() as session:
        assert session.query(SqlIssue).count() == 1
