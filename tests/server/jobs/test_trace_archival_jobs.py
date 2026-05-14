from unittest.mock import MagicMock, patch

import mlflow.tracing._trace_archival_service as trace_archival_service_module
import mlflow.tracing.trace_archival_config as trace_archival_config_module
from mlflow.entities import Workspace
from mlflow.entities.workspace import TraceArchivalConfig
from mlflow.environment_variables import (
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_TRACE_ARCHIVAL_CONFIG,
)
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.utils import register_periodic_tasks
from mlflow.store.workspace.abstract_store import ResolvedTraceArchivalConfig
from mlflow.tracing._trace_archival_service import run_trace_archival_scheduler
from mlflow.utils.workspace_context import get_request_workspace


def _configure_trace_archival_scheduler(
    monkeypatch,
    tmp_path,
    *,
    workspaces_enabled: bool,
    enabled: bool = True,
    interval_seconds: int = 1,
    max_traces_per_pass: int | None = None,
):
    config_path = tmp_path / "trace-archival.yaml"
    lines = [
        "trace_archival:",
        f"  enabled: {'true' if enabled else 'false'}",
        "  location: s3://archive/default",
        "  retention: 30d",
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


def test_trace_archival_scheduler_runs_per_workspace(monkeypatch, tmp_path):
    _configure_trace_archival_scheduler(monkeypatch, tmp_path, workspaces_enabled=True)

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
