from unittest.mock import patch

import pytest

import mlflow.tracing.trace_archival_config as trace_archival_config_module
from mlflow.environment_variables import MLFLOW_TRACE_ARCHIVAL_CONFIG
from mlflow.exceptions import MlflowException
from mlflow.tracing.trace_archival_config import get_trace_archival_server_config


@pytest.fixture(autouse=True)
def reset_trace_archival_server_config_cache(monkeypatch):
    monkeypatch.setattr(trace_archival_config_module, "_TRACE_ARCHIVAL_SERVER_CONFIG_CACHE", None)


def _write_trace_archival_config_lines(tmp_path, lines):
    config_path = tmp_path / "trace-archival.yaml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _write_trace_archival_config(tmp_path, *, retention="30d"):
    archive_path = tmp_path / "archive"
    archive_path.mkdir(exist_ok=True)
    return _write_trace_archival_config_lines(
        tmp_path,
        [
            "trace_archival:",
            "  enabled: true",
            f"  location: {archive_path.as_uri()}",
            f"  retention: {retention}",
        ],
    )


def test_get_trace_archival_server_config_rejects_null_location(monkeypatch, tmp_path):
    config_path = _write_trace_archival_config_lines(
        tmp_path,
        [
            "trace_archival:",
            "  enabled: true",
            "  location:",
            "  retention: 30d",
        ],
    )
    monkeypatch.setenv(MLFLOW_TRACE_ARCHIVAL_CONFIG.name, str(config_path))

    with pytest.raises(MlflowException, match="trace_archival.location") as exc_info:
        get_trace_archival_server_config()

    assert "Expected a URI string." in exc_info.value.message


def test_get_trace_archival_server_config_rejects_null_retention(monkeypatch, tmp_path):
    config_path = _write_trace_archival_config_lines(
        tmp_path,
        [
            "trace_archival:",
            "  enabled: true",
            f"  location: {(tmp_path / 'archive').as_uri()}",
            "  retention:",
        ],
    )
    monkeypatch.setenv(MLFLOW_TRACE_ARCHIVAL_CONFIG.name, str(config_path))

    with pytest.raises(MlflowException, match="trace_archival.retention") as exc_info:
        get_trace_archival_server_config()

    assert "Expected a duration in the form" in exc_info.value.message


def test_get_trace_archival_server_config_uses_cached_value_before_expiry(monkeypatch, tmp_path):
    config_path = _write_trace_archival_config(tmp_path, retention="30d")
    monkeypatch.setenv(MLFLOW_TRACE_ARCHIVAL_CONFIG.name, str(config_path))

    with patch.object(trace_archival_config_module.time, "monotonic", side_effect=[10.0, 12.0]):
        first = get_trace_archival_server_config()
        config_path.write_text(
            "\n".join([
                "trace_archival:",
                "  enabled: true",
                f"  location: {(tmp_path / 'archive').as_uri()}",
                "  retention: 7d",
            ])
            + "\n",
            encoding="utf-8",
        )
        second = get_trace_archival_server_config()

    assert first is not None
    assert second is not None
    assert first.retention == "30d"
    assert second.retention == "30d"


def test_get_trace_archival_server_config_reloads_and_logs_changes(monkeypatch, tmp_path):
    config_path = _write_trace_archival_config(tmp_path, retention="30d")
    monkeypatch.setenv(MLFLOW_TRACE_ARCHIVAL_CONFIG.name, str(config_path))

    with (
        patch.object(trace_archival_config_module.time, "monotonic", side_effect=[10.0, 16.0]),
        patch.object(trace_archival_config_module, "_logger") as mock_logger,
    ):
        first = get_trace_archival_server_config()
        config_path.write_text(
            "\n".join([
                "trace_archival:",
                "  enabled: true",
                f"  location: {(tmp_path / 'archive').as_uri()}",
                "  retention: 7d",
                "  max_traces_per_pass: 25",
            ])
            + "\n",
            encoding="utf-8",
        )
        second = get_trace_archival_server_config()

    assert first is not None
    assert second is not None
    assert first.retention == "30d"
    assert second.retention == "7d"
    assert second.max_traces_per_pass == 25
    mock_logger.info.assert_called_once_with(
        "Trace archival config changed; refreshed cached server settings."
    )


def test_get_trace_archival_server_config_keeps_last_good_value_on_refresh_failure(
    monkeypatch, tmp_path
):
    config_path = _write_trace_archival_config(tmp_path, retention="30d")
    monkeypatch.setenv(MLFLOW_TRACE_ARCHIVAL_CONFIG.name, str(config_path))

    with (
        patch.object(trace_archival_config_module.time, "monotonic", side_effect=[10.0, 16.0]),
        patch.object(trace_archival_config_module, "_logger") as mock_logger,
    ):
        first = get_trace_archival_server_config()
        config_path.write_text("trace_archival: [\n", encoding="utf-8")
        second = get_trace_archival_server_config()

    assert first is not None
    assert second is not None
    assert second == first
    mock_logger.warning.assert_called_once_with(
        "Failed to refresh trace archival config; continuing to use the last valid config.",
        exc_info=True,
    )
