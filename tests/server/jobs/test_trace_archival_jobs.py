import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as _OTelResource
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow.tracing.trace_archival_config as trace_archival_config_module
import mlflow.tracing.trace_archival_service as trace_archival_service_module
from mlflow.entities import ExperimentTag, Workspace, trace_location
from mlflow.entities.span import create_mlflow_span
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_state import TraceState
from mlflow.entities.workspace import TraceArchivalConfig
from mlflow.environment_variables import (
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_TRACE_ARCHIVAL_CONFIG,
    MLFLOW_WORKSPACE,
)
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.utils import register_periodic_tasks
from mlflow.store.tracking.dbmodels.models import SqlSpan
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.store.workspace.abstract_store import ResolvedTraceArchivalConfig
from mlflow.tracing.constant import SpansLocation, TraceExperimentTagKey, TraceTagKey
from mlflow.tracing.otel.otel_archival import TRACE_ARCHIVAL_FILENAME
from mlflow.tracing.trace_archival_service import run_trace_archival_scheduler
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.uri import append_to_uri_path
from mlflow.utils.workspace_context import WorkspaceContext, get_request_workspace
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


def _configure_trace_archival_scheduler(
    monkeypatch,
    tmp_path,
    *,
    workspaces_enabled: bool,
    enabled: bool = True,
    location: str = "s3://archive/default",
    retention: str = "30d",
    interval_seconds: int = 1,
    max_traces_per_pass: int | None = None,
):
    config_path = tmp_path / "trace-archival.yaml"
    lines = [
        "trace_archival:",
        f"  enabled: {'true' if enabled else 'false'}",
        f"  location: {location}",
        f"  retention: {retention}",
        f"  interval_seconds: {interval_seconds}",
    ]
    if max_traces_per_pass is not None:
        lines.append(f"  max_traces_per_pass: {max_traces_per_pass}")

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv(MLFLOW_TRACE_ARCHIVAL_CONFIG.name, str(config_path))
    monkeypatch.setenv(
        MLFLOW_ENABLE_WORKSPACES.name,
        "true" if workspaces_enabled else "false",
    )
    monkeypatch.setattr(
        trace_archival_service_module,
        "_TRACE_ARCHIVAL_SCHEDULER_LAST_RUN_MONOTONIC",
        0.0,
    )
    monkeypatch.setattr(trace_archival_config_module, "_TRACE_ARCHIVAL_SERVER_CONFIG_CACHE", None)


@contextmanager
def _create_tracking_store(tmp_path: Path, *, workspaces_enabled: bool):
    store_cls = WorkspaceAwareSqlAlchemyStore if workspaces_enabled else SqlAlchemyStore
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    store = store_cls(f"sqlite:///{tmp_path / 'mlflow.db'}", artifact_root.as_uri())
    try:
        yield store
    finally:
        store.engine.dispose()


def _create_mock_span_context(trace_id_num: int = 12345, span_id_num: int = 111):
    context = MagicMock()
    context.trace_id = trace_id_num
    context.span_id = span_id_num
    context.is_remote = False
    context.trace_flags = trace_api.TraceFlags(1)
    context.trace_state = trace_api.TraceState()
    return context


def _create_test_span(
    trace_id: str,
    *,
    name: str = "test_span",
    span_id: int = 111,
    start_ns: int = 1_000_000_000,
    end_ns: int = 2_000_000_000,
    span_type: str = "LLM",
):
    otel_span = OTelReadableSpan(
        name=name,
        context=_create_mock_span_context(span_id_num=span_id),
        parent=None,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanType": json.dumps(span_type, cls=TraceJSONEncoder),
        },
        start_time=start_ns,
        end_time=end_ns,
        status=trace_api.Status(trace_api.StatusCode.UNSET),
        resource=_OTelResource.get_empty(),
    )
    return create_mlflow_span(otel_span, trace_id, span_type)


def _create_trace(
    store: SqlAlchemyStore,
    trace_id: str,
    *,
    experiment_id: str,
    request_time: int,
) -> TraceInfo:
    return store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
            request_time=request_time,
            execution_duration=0,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id=f"{trace_id}-client",
        )
    )


def _workspace_context(workspaces_enabled: bool):
    return WorkspaceContext(DEFAULT_WORKSPACE_NAME) if workspaces_enabled else nullcontext()


def _get_archive_payload_path(archive_uri: str) -> Path:
    return Path(local_file_uri_to_path(archive_uri)) / TRACE_ARCHIVAL_FILENAME


def test_trace_archival_scheduler_runs_per_workspace(monkeypatch, tmp_path):
    _configure_trace_archival_scheduler(monkeypatch, tmp_path, workspaces_enabled=True)
    monkeypatch.delenv(MLFLOW_WORKSPACE.name, raising=False)

    mock_tracking_store = MagicMock()
    mock_workspace_store = MagicMock()
    mock_workspace_store.list_workspaces.return_value = [
        Workspace(name="team-a"),
        Workspace(name="team-b"),
    ]
    workspace_calls = []

    def archive_traces(**kwargs):
        workspace_calls.append((get_request_workspace(), kwargs))
        return 1

    mock_tracking_store.resolve_trace_archival_config.return_value = ResolvedTraceArchivalConfig(
        config=TraceArchivalConfig(location="s3://archive/default", retention="30d"),
        append_workspace_prefix=True,
    )

    with (
        patch("mlflow.server.handlers._get_tracking_store", return_value=mock_tracking_store),
        patch(
            "mlflow.server.workspace_helpers._get_workspace_store",
            return_value=mock_workspace_store,
        ),
        patch.object(
            trace_archival_service_module.random,
            "shuffle",
            side_effect=lambda workspaces: workspaces.reverse(),
        ) as shuffle_mock,
    ):
        mock_tracking_store.archive_traces.side_effect = archive_traces
        archived = run_trace_archival_scheduler()

    assert archived == 2
    shuffle_mock.assert_called_once()
    assert [workspace for workspace, _ in workspace_calls] == ["team-b", "team-a"]
    assert MLFLOW_WORKSPACE.get_raw() is None
    for workspace, kwargs in workspace_calls:
        assert (
            kwargs["resolved_trace_archival_location"]
            == f"s3://archive/default/workspaces/{workspace}"
        )
        assert kwargs["broader_retention"] == "30d"


def test_trace_archival_scheduler_skips_unsupported_workspace_and_continues(monkeypatch, tmp_path):
    _configure_trace_archival_scheduler(monkeypatch, tmp_path, workspaces_enabled=True)

    mock_tracking_store = MagicMock()
    mock_workspace_store = MagicMock()
    mock_workspace_store.list_workspaces.return_value = [
        Workspace(name="team-a"),
        Workspace(name="team-b"),
    ]
    workspace_calls = []

    def resolve_trace_archival_config(tracking_store, **kwargs):
        workspace = get_request_workspace()
        workspace_calls.append(workspace)
        if workspace == "team-a":
            raise MlflowException.invalid_parameter_value("unsupported archival destination")
        return TraceArchivalConfig(
            location=f"s3://archive/default/workspaces/{workspace}",
            retention="30d",
        )

    with (
        patch("mlflow.server.handlers._get_tracking_store", return_value=mock_tracking_store),
        patch(
            "mlflow.server.workspace_helpers._get_workspace_store",
            return_value=mock_workspace_store,
        ),
        patch.object(
            trace_archival_service_module,
            "_resolve_scheduler_trace_archival_config",
            side_effect=resolve_trace_archival_config,
        ),
        patch.object(
            trace_archival_service_module.random,
            "shuffle",
            side_effect=lambda workspaces: workspaces.reverse(),
        ) as shuffle_mock,
    ):
        mock_tracking_store.archive_traces.return_value = 2
        archived = run_trace_archival_scheduler()

    assert archived == 2
    shuffle_mock.assert_called_once()
    assert workspace_calls == ["team-b", "team-a"]
    mock_tracking_store.archive_traces.assert_called_once_with(
        resolved_trace_archival_location="s3://archive/default/workspaces/team-b",
        broader_retention="30d",
        long_retention_allowlist=set(),
        max_traces_per_pass=None,
    )


def test_trace_archival_scheduler_respects_interval(monkeypatch, tmp_path):
    _configure_trace_archival_scheduler(
        monkeypatch,
        tmp_path,
        workspaces_enabled=False,
        interval_seconds=60,
    )

    mock_tracking_store = MagicMock()
    mock_tracking_store.resolve_trace_archival_config.return_value = ResolvedTraceArchivalConfig(
        config=TraceArchivalConfig(location="s3://archive/default", retention="30d"),
        append_workspace_prefix=False,
    )
    with (
        patch("mlflow.server.handlers._get_tracking_store", return_value=mock_tracking_store),
        patch.object(
            trace_archival_service_module.time,
            "monotonic",
            side_effect=[100.0, 100.1, 100.2, 100.3, 110.0, 120.0],
        ),
    ):
        mock_tracking_store.archive_traces.return_value = 3
        first = run_trace_archival_scheduler()
        second = run_trace_archival_scheduler()

    assert first == 3
    assert second == 0
    mock_tracking_store.archive_traces.assert_called_once()


def test_trace_archival_scheduler_returns_zero_when_disabled(monkeypatch, tmp_path):
    _configure_trace_archival_scheduler(
        monkeypatch,
        tmp_path,
        workspaces_enabled=False,
        enabled=False,
    )

    with patch("mlflow.server.handlers._get_tracking_store") as mock_get_tracking_store:
        archived = run_trace_archival_scheduler()

    assert archived == 0
    mock_get_tracking_store.assert_not_called()


def test_trace_archival_scheduler_passes_max_traces_per_pass(monkeypatch, tmp_path):
    _configure_trace_archival_scheduler(
        monkeypatch,
        tmp_path,
        workspaces_enabled=False,
        max_traces_per_pass=7,
    )

    mock_tracking_store = MagicMock()
    mock_tracking_store.resolve_trace_archival_config.return_value = ResolvedTraceArchivalConfig(
        config=TraceArchivalConfig(location="s3://archive/default", retention="30d"),
        append_workspace_prefix=False,
    )
    with patch("mlflow.server.handlers._get_tracking_store", return_value=mock_tracking_store):
        mock_tracking_store.archive_traces.return_value = 3
        archived = run_trace_archival_scheduler()

    assert archived == 3
    mock_tracking_store.archive_traces.assert_called_once_with(
        resolved_trace_archival_location="s3://archive/default",
        broader_retention="30d",
        long_retention_allowlist=set(),
        max_traces_per_pass=7,
    )


def test_trace_archival_scheduler_shares_pass_budget_across_workspaces(monkeypatch, tmp_path):
    _configure_trace_archival_scheduler(
        monkeypatch,
        tmp_path,
        workspaces_enabled=True,
        max_traces_per_pass=1,
    )

    mock_tracking_store = MagicMock()
    mock_workspace_store = MagicMock()
    mock_workspace_store.list_workspaces.return_value = [
        Workspace(name="team-a"),
        Workspace(name="team-b"),
    ]
    mock_tracking_store.resolve_trace_archival_config.return_value = ResolvedTraceArchivalConfig(
        config=TraceArchivalConfig(location="s3://archive/default", retention="30d"),
        append_workspace_prefix=True,
    )

    with (
        patch("mlflow.server.handlers._get_tracking_store", return_value=mock_tracking_store),
        patch(
            "mlflow.server.workspace_helpers._get_workspace_store",
            return_value=mock_workspace_store,
        ),
        patch.object(
            trace_archival_service_module.random,
            "shuffle",
            side_effect=lambda workspaces: workspaces.reverse(),
        ) as shuffle_mock,
    ):
        mock_tracking_store.archive_traces.return_value = 1
        archived = run_trace_archival_scheduler()

    assert archived == 1
    shuffle_mock.assert_called_once()
    mock_tracking_store.archive_traces.assert_called_once_with(
        resolved_trace_archival_location="s3://archive/default/workspaces/team-b",
        broader_retention="30d",
        long_retention_allowlist=set(),
        max_traces_per_pass=1,
    )


def test_trace_archival_scheduler_archives_real_store_traces(monkeypatch, tmp_path):
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _configure_trace_archival_scheduler(
        monkeypatch,
        tmp_path,
        workspaces_enabled=False,
        location=archive_root.as_uri(),
        retention="1d",
    )

    now_millis = 10 * 24 * 60 * 60 * 1000
    old_trace_id = "tr-archive-old"
    fresh_trace_id = "tr-archive-fresh"
    old_request_time = now_millis - 3 * 24 * 60 * 60 * 1000
    fresh_request_time = now_millis - 60 * 60 * 1000

    with _create_tracking_store(tmp_path, workspaces_enabled=False) as store:
        exp_id = store.create_experiment("archive-db-backed")
        _create_trace(
            store,
            old_trace_id,
            experiment_id=exp_id,
            request_time=old_request_time,
        )
        _create_trace(
            store,
            fresh_trace_id,
            experiment_id=exp_id,
            request_time=fresh_request_time,
        )
        store.log_spans(
            exp_id,
            [
                _create_test_span(
                    old_trace_id,
                    span_id=111,
                    start_ns=old_request_time * 1_000_000,
                    end_ns=(old_request_time + 1_000) * 1_000_000,
                )
            ],
        )
        store.log_spans(
            exp_id,
            [
                _create_test_span(
                    fresh_trace_id,
                    span_id=222,
                    start_ns=fresh_request_time * 1_000_000,
                    end_ns=(fresh_request_time + 1_000) * 1_000_000,
                )
            ],
        )

        with (
            patch("mlflow.server.handlers._get_tracking_store", return_value=store),
            patch.object(store, "_get_archive_traces_now_millis", return_value=now_millis),
        ):
            archived = run_trace_archival_scheduler()

        assert archived == 1

        old_trace_info = store.get_trace_info(old_trace_id)
        fresh_trace_info = store.get_trace_info(fresh_trace_id)
        expected_archive_uri = append_to_uri_path(
            archive_root.as_uri(),
            exp_id,
            SqlAlchemyStore.TRACE_FOLDER_NAME,
            old_trace_id,
            SqlAlchemyStore.ARTIFACTS_FOLDER_NAME,
        )

        assert old_trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.ARCHIVE_REPO.value
        assert old_trace_info.tags[TraceTagKey.ARCHIVE_LOCATION] == expected_archive_uri
        assert TraceTagKey.ARCHIVE_LOCATION not in fresh_trace_info.tags
        assert _get_archive_payload_path(
            old_trace_info.tags[TraceTagKey.ARCHIVE_LOCATION]
        ).is_file()

        with store.ManagedSessionMaker() as session:
            archived_span = session.query(SqlSpan).filter(SqlSpan.trace_id == old_trace_id).one()
            fresh_span = session.query(SqlSpan).filter(SqlSpan.trace_id == fresh_trace_id).one()
            assert archived_span.content == ""
            assert fresh_span.content != ""

        assert store.get_trace(old_trace_id).data.spans[0].name == "test_span"
        batched_traces = store.batch_get_traces([old_trace_id, fresh_trace_id])
        assert [trace.info.trace_id for trace in batched_traces] == [old_trace_id, fresh_trace_id]
        assert [trace.data.spans[0].name for trace in batched_traces] == ["test_span", "test_span"]


def test_trace_archival_scheduler_honors_workspace_archive_location(monkeypatch, tmp_path):
    server_archive_root = tmp_path / "server-archive"
    workspace_archive_root = tmp_path / "workspace-archive"
    server_archive_root.mkdir()
    workspace_archive_root.mkdir()
    _configure_trace_archival_scheduler(
        monkeypatch,
        tmp_path,
        workspaces_enabled=True,
        location=server_archive_root.as_uri(),
        retention="1d",
    )

    now_millis = 12 * 24 * 60 * 60 * 1000
    trace_id = "tr-workspace-archive"
    request_time = now_millis - 3 * 24 * 60 * 60 * 1000

    with _create_tracking_store(tmp_path, workspaces_enabled=True) as store:
        workspace_store = store._get_workspace_provider_instance()
        with _workspace_context(workspaces_enabled=True):
            workspace_store.update_workspace(
                Workspace(
                    name=DEFAULT_WORKSPACE_NAME,
                    trace_archival_location=workspace_archive_root.as_uri(),
                )
            )
            exp_id = store.create_experiment("archive-workspace-override")
            _create_trace(
                store,
                trace_id,
                experiment_id=exp_id,
                request_time=request_time,
            )
            store.log_spans(
                exp_id,
                [
                    _create_test_span(
                        trace_id,
                        span_id=811,
                        start_ns=request_time * 1_000_000,
                        end_ns=(request_time + 1_000) * 1_000_000,
                    )
                ],
            )

        with (
            patch("mlflow.server.handlers._get_tracking_store", return_value=store),
            patch(
                "mlflow.server.workspace_helpers._get_workspace_store",
                return_value=workspace_store,
            ),
            patch.object(store, "_get_archive_traces_now_millis", return_value=now_millis),
        ):
            archived = run_trace_archival_scheduler()

        assert archived == 1

        with _workspace_context(workspaces_enabled=True):
            trace_info = store.get_trace_info(trace_id)

        expected_archive_uri = append_to_uri_path(
            workspace_archive_root.as_uri(),
            exp_id,
            SqlAlchemyStore.TRACE_FOLDER_NAME,
            trace_id,
            SqlAlchemyStore.ARTIFACTS_FOLDER_NAME,
        )

        assert trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.ARCHIVE_REPO.value
        assert trace_info.tags[TraceTagKey.ARCHIVE_LOCATION] == expected_archive_uri
        assert _get_archive_payload_path(trace_info.tags[TraceTagKey.ARCHIVE_LOCATION]).is_file()
        assert not any(server_archive_root.iterdir())


def test_trace_archival_scheduler_processes_archive_now_with_real_store(monkeypatch, tmp_path):
    archive_root = tmp_path / "archive-now"
    archive_root.mkdir()
    _configure_trace_archival_scheduler(
        monkeypatch,
        tmp_path,
        workspaces_enabled=False,
        location=archive_root.as_uri(),
        retention="30d",
    )

    now_millis = 14 * 24 * 60 * 60 * 1000
    old_trace_id = "tr-archive-now-old"
    fresh_trace_id = "tr-archive-now-fresh"
    old_request_time = now_millis - 2 * 24 * 60 * 60 * 1000
    fresh_request_time = now_millis - 2 * 60 * 60 * 1000

    with _create_tracking_store(tmp_path, workspaces_enabled=False) as store:
        exp_id = store.create_experiment("archive-now-scheduler")
        store.set_experiment_tag(
            exp_id,
            ExperimentTag(TraceExperimentTagKey.ARCHIVE_NOW, json.dumps({"older_than": "1d"})),
        )
        _create_trace(
            store,
            old_trace_id,
            experiment_id=exp_id,
            request_time=old_request_time,
        )
        _create_trace(
            store,
            fresh_trace_id,
            experiment_id=exp_id,
            request_time=fresh_request_time,
        )
        store.log_spans(
            exp_id,
            [
                _create_test_span(
                    old_trace_id,
                    span_id=311,
                    start_ns=old_request_time * 1_000_000,
                    end_ns=(old_request_time + 1_000) * 1_000_000,
                )
            ],
        )
        store.log_spans(
            exp_id,
            [
                _create_test_span(
                    fresh_trace_id,
                    span_id=312,
                    start_ns=fresh_request_time * 1_000_000,
                    end_ns=(fresh_request_time + 1_000) * 1_000_000,
                )
            ],
        )

        with (
            patch("mlflow.server.handlers._get_tracking_store", return_value=store),
            patch.object(store, "_get_archive_traces_now_millis", return_value=now_millis),
        ):
            archived = run_trace_archival_scheduler()

        assert archived == 1

        old_trace_info = store.get_trace_info(old_trace_id)
        fresh_trace_info = store.get_trace_info(fresh_trace_id)
        assert old_trace_info.tags[TraceTagKey.SPANS_LOCATION] == SpansLocation.ARCHIVE_REPO.value
        assert _get_archive_payload_path(
            old_trace_info.tags[TraceTagKey.ARCHIVE_LOCATION]
        ).is_file()
        assert (
            fresh_trace_info.tags.get(
                TraceTagKey.SPANS_LOCATION, SpansLocation.TRACKING_STORE.value
            )
            == SpansLocation.TRACKING_STORE.value
        )
        assert TraceExperimentTagKey.ARCHIVE_NOW not in store.get_experiment(exp_id).tags
        assert store.get_trace(old_trace_id).data.spans[0].name == "test_span"


class _RecordingHuey:
    def __init__(self):
        self.periodic_task_names = []

    def periodic_task(self, *_args, **_kwargs):
        def decorator(fn):
            self.periodic_task_names.append(fn.__name__)
            return fn

        return decorator

    def lock_task(self, *_args, **_kwargs):
        def decorator(fn):
            return fn

        return decorator


def test_register_periodic_tasks_includes_trace_archival_when_unconfigured(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACE_ARCHIVAL_CONFIG.name, raising=False)

    huey = _RecordingHuey()

    register_periodic_tasks(huey)

    assert "online_scoring_scheduler" in huey.periodic_task_names
    assert "trace_archival_scheduler" in huey.periodic_task_names


def test_register_periodic_tasks_includes_trace_archival_when_config_invalid(monkeypatch, tmp_path):
    config_path = tmp_path / "trace-archival.yaml"
    config_path.write_text("trace_archival: [\n", encoding="utf-8")
    monkeypatch.setenv(MLFLOW_TRACE_ARCHIVAL_CONFIG.name, str(config_path))

    huey = _RecordingHuey()

    register_periodic_tasks(huey)

    assert "online_scoring_scheduler" in huey.periodic_task_names
    assert "trace_archival_scheduler" in huey.periodic_task_names


def test_trace_archival_scheduler_logs_warning_when_config_invalid(monkeypatch, tmp_path):
    config_path = tmp_path / "trace-archival.yaml"
    config_path.write_text("trace_archival: [\n", encoding="utf-8")
    monkeypatch.setenv(MLFLOW_TRACE_ARCHIVAL_CONFIG.name, str(config_path))
    monkeypatch.setattr(trace_archival_config_module, "_TRACE_ARCHIVAL_SERVER_CONFIG_CACHE", None)

    with patch.object(trace_archival_service_module, "_logger") as mock_logger:
        archived = run_trace_archival_scheduler()

    assert archived == 0
    mock_logger.warning.assert_called_once_with(
        "Ignoring invalid trace archival scheduler configuration.",
        exc_info=True,
    )


def test_register_periodic_tasks_includes_trace_archival_when_configured(monkeypatch, tmp_path):
    _configure_trace_archival_scheduler(monkeypatch, tmp_path, workspaces_enabled=False)

    huey = _RecordingHuey()

    register_periodic_tasks(huey)

    assert "online_scoring_scheduler" in huey.periodic_task_names
    assert "trace_archival_scheduler" in huey.periodic_task_names
