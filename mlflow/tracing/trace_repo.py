"""
Trace repository orchestrator.

Coordinates between the tracking store (DB-only primitives) and artifact
repositories for trace archival, retrieval, and deletion.  Callers (handlers,
tracing client) should use these functions instead of invoking the store's
archival methods directly.

Store-type dispatch:
  * Local stores (SqlAlchemyStore) expose DB primitives
    (``collect_archive_candidates``, ``read_trace_for_archive``, …).  The
    orchestrator uses those primitives together with the artifact repo layer.
  * REST stores proxy every request to the tracking server, so the
    orchestrator delegates to the store's existing ``archive_traces`` /
    ``delete_traces`` methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from mlflow.entities.span import Span
from mlflow.entities.trace_info import TraceInfo
from mlflow.exceptions import MlflowException, MlflowTraceDataNotFound, MlflowTracingException
from mlflow.utils.time import get_current_time_millis

_logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 50


def _store_has_local_primitives(store) -> bool:
    return hasattr(store, "ManagedSessionMaker")


# ---------------------------------------------------------------------------
# Data class returned by store.read_trace_for_archive
# ---------------------------------------------------------------------------


@dataclass
class TraceArchiveData:
    """Lightweight container holding everything needed to write a trace archive."""

    trace_id: str
    experiment_id: int
    spans: list[Span]
    artifact_uri: str


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


def archive_traces(
    store,
    *,
    workspace: str | None = None,
    older_than: timedelta | None = None,
    trace_ids: list[str] | None = None,
    experiment_id: str | None = None,
    filter_string: str | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> int:
    """Archive traces from the tracking store to the archive repository.

    For local stores, orchestrates: candidate selection -> span read ->
    artifact write -> DB mutation.  For REST stores, delegates to
    ``store.archive_traces()``.
    """
    _validate_archive_params(older_than, trace_ids)

    if not _store_has_local_primitives(store):
        return store.archive_traces(
            workspace=workspace,
            older_than=older_than,
            trace_ids=trace_ids,
            experiment_id=experiment_id,
            filter_string=filter_string,
        )

    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
    from mlflow.tracing.constant import SpansLocation

    cutoff_ms = (
        get_current_time_millis() - int(older_than.total_seconds() * 1000)
        if older_than is not None
        else None
    )

    candidate_rows = store.collect_archive_candidates(
        workspace=workspace,
        experiment_id=experiment_id,
        trace_ids=trace_ids,
        cutoff_ms=cutoff_ms,
        filter_string=filter_string,
    )
    if not candidate_rows:
        return 0

    archived_count = 0
    total = len(candidate_rows)
    for batch_start in range(0, total, batch_size):
        batch = candidate_rows[batch_start : batch_start + batch_size]
        for tid, exp_id in batch:
            if exp_id is None:
                continue
            try:
                archive_data = store.read_trace_for_archive(tid, exp_id)
                if archive_data is None:
                    continue

                artifact_repo = get_artifact_repository(archive_data.artifact_uri)
                artifact_repo.upload_trace_data(
                    spans=archive_data.spans,
                    spans_location=SpansLocation.ARCHIVE_REPO,
                )

                store.mark_trace_archived(tid, archive_data.artifact_uri)
                archived_count += 1
            except Exception as e:
                _logger.warning("Failed to archive trace %s: %s. Skipping.", tid, e)
                continue

        _logger.info(
            "Archive progress: %d / %d candidates processed (%d archived so far).",
            min(batch_start + len(batch), total),
            total,
            archived_count,
        )

    return archived_count


# ---------------------------------------------------------------------------
# Read archived spans
# ---------------------------------------------------------------------------


def load_archived_spans(store, trace_info: TraceInfo) -> list[Span]:
    """Load spans from the archive repository for an archived trace."""
    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
    from mlflow.tracing.constant import SpansLocation

    uri = store.get_trace_repository_artifact_uri(trace_info)
    if uri is None:
        raise MlflowTracingException(
            "Trace data not found in archive repository: "
            f"no artifact URI for trace {trace_info.trace_id}"
        )
    try:
        artifact_repo = get_artifact_repository(uri)
        return artifact_repo.download_trace_data(spans_location=SpansLocation.ARCHIVE_REPO)
    except MlflowTraceDataNotFound as e:
        raise MlflowTracingException("Trace data not found in archive repository") from e


# ---------------------------------------------------------------------------
# Delete (with artifact cleanup)
# ---------------------------------------------------------------------------


def delete_traces(
    store,
    experiment_id: str,
    max_timestamp_millis: int | None = None,
    max_traces: int | None = None,
    trace_ids: list[str] | None = None,
) -> int:
    """Delete traces, cleaning up archived artifacts first.

    For local stores: finds traces in ARCHIVE_REPO, best-effort deletes their
    artifacts, then removes all matching DB rows.  For REST stores, delegates
    to ``store.delete_traces()``.
    """
    if not _store_has_local_primitives(store):
        return store.delete_traces(
            experiment_id=experiment_id,
            max_timestamp_millis=max_timestamp_millis,
            max_traces=max_traces,
            trace_ids=trace_ids,
        )

    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
    from mlflow.tracing.otel.otel_archival import TRACE_ARCHIVAL_ARTIFACT_PATH

    archived_uris = store.find_archived_trace_uris(
        experiment_id=experiment_id,
        max_timestamp_millis=max_timestamp_millis,
        max_traces=max_traces,
        trace_ids=trace_ids,
    )
    for tid, uri in archived_uris.items():
        try:
            artifact_repo = get_artifact_repository(uri)
            artifact_repo.delete_artifacts(artifact_path=TRACE_ARCHIVAL_ARTIFACT_PATH)
        except Exception as e:
            _logger.warning(
                "Could not delete archive repository file for trace %s: %s. "
                "DB records will still be removed; artifact may be orphaned.",
                tid,
                e,
            )

    return store.delete_trace_rows(
        experiment_id=experiment_id,
        max_timestamp_millis=max_timestamp_millis,
        max_traces=max_traces,
        trace_ids=trace_ids,
    )


# ---------------------------------------------------------------------------
# Validation (moved from AbstractTrackingStore.archive_traces)
# ---------------------------------------------------------------------------


def _validate_archive_params(
    older_than: timedelta | None,
    trace_ids: list[str] | None,
) -> None:
    has_trace_ids = trace_ids is not None and len(trace_ids) > 0
    if not has_trace_ids and older_than is None:
        raise MlflowException.invalid_parameter_value(
            "Specify at least one of older_than or trace_ids."
        )
    if older_than is not None and older_than.total_seconds() <= 0:
        raise MlflowException.invalid_parameter_value(
            f"older_than must be positive, got {older_than}."
        )
