from __future__ import annotations

from unittest import mock

from mlflow.tracing.trace_repo import (
    TraceArchiveData,
    archive_traces,
    load_archived_spans,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_local_store(**overrides):
    store = mock.MagicMock()
    store.ManagedSessionMaker = mock.MagicMock()
    store.collect_archive_candidates = mock.MagicMock(return_value=[])
    store.read_trace_for_archive = mock.MagicMock(return_value=None)
    store.mark_trace_archived = mock.MagicMock()
    store.get_trace_repository_artifact_uri = mock.MagicMock(return_value=None)
    for k, v in overrides.items():
        setattr(store, k, v)
    return store


def _make_archive_data(trace_id="t1", experiment_id=1):
    return TraceArchiveData(
        trace_id=trace_id,
        experiment_id=experiment_id,
        spans=[mock.MagicMock()],
        artifact_uri=f"file:///repo/{experiment_id}/traces/{trace_id}",
    )


# ---------------------------------------------------------------------------
# archive_traces - verifies unified upload_trace_data API
# ---------------------------------------------------------------------------


def test_archive_candidates_and_writes_artifacts():
    archive_data = _make_archive_data("t1", 1)
    store = _make_local_store(
        collect_archive_candidates=mock.MagicMock(return_value=[("t1", 1)]),
        read_trace_for_archive=mock.MagicMock(return_value=archive_data),
    )
    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"
    ) as mock_get_repo:
        result = archive_traces(store, trace_ids=["t1"])
    assert result == 1
    store.read_trace_for_archive.assert_called_once_with("t1", 1)
    store.mark_trace_archived.assert_called_once_with("t1", archive_data.artifact_uri)
    mock_get_repo.assert_called_once_with(archive_data.artifact_uri)
    mock_get_repo.return_value.upload_trace_data.assert_called_once()


def test_archive_skips_on_exception():
    store = _make_local_store(
        collect_archive_candidates=mock.MagicMock(return_value=[("t1", 1), ("t2", 2)]),
        read_trace_for_archive=mock.MagicMock(
            side_effect=[RuntimeError("boom"), _make_archive_data("t2", 2)]
        ),
    )
    with mock.patch("mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"):
        result = archive_traces(store, trace_ids=["t1"])
    assert result == 1
    store.mark_trace_archived.assert_called_once()


def test_archive_batch_processing():
    candidates = [(f"t{i}", i) for i in range(5)]
    store = _make_local_store(
        collect_archive_candidates=mock.MagicMock(return_value=candidates),
        read_trace_for_archive=mock.MagicMock(
            side_effect=[_make_archive_data(f"t{i}", i) for i in range(5)]
        ),
    )
    with mock.patch("mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"):
        result = archive_traces(store, trace_ids=["t0"], batch_size=2)
    assert result == 5
    assert store.mark_trace_archived.call_count == 5


# ---------------------------------------------------------------------------
# load_archived_spans - verifies unified download_trace_data API
# ---------------------------------------------------------------------------


def test_load_archived_loads_from_artifact_repo():
    from mlflow.tracing.constant import SpansLocation

    spans = [mock.MagicMock()]
    store = _make_local_store(
        get_trace_repository_artifact_uri=mock.MagicMock(return_value="file:///repo/t1"),
    )
    trace_info = mock.MagicMock()
    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"
    ) as mock_get_repo:
        mock_get_repo.return_value.download_trace_data.return_value = spans
        result = load_archived_spans(store, trace_info)
    assert result == spans
    mock_get_repo.assert_called_once_with("file:///repo/t1")
    mock_get_repo.return_value.download_trace_data.assert_called_once_with(
        spans_location=SpansLocation.ARCHIVE_REPO
    )
