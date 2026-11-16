"""Trace loading utilities for online scoring."""

import logging

from mlflow.entities import Trace, TraceData, TraceInfo
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import SpansLocation, TraceTagKey
from mlflow.tracing.utils.artifact_utils import get_artifact_uri_for_trace

_logger = logging.getLogger(__name__)

# Trace IDs per batch_get_traces call.
_FETCH_CHUNK_SIZE = 200


def trace_end_time_ms(trace_info: TraceInfo) -> int:
    """
    The trace's end time as the store computes it for ``trace.end_time_ms``: start plus duration.

    For an in-progress trace this is the end of its latest exported span, so it moves forward as
    spans arrive and reflects when the trace last made progress.
    """
    return trace_info.timestamp_ms + (trace_info.execution_time_ms or 0)


def trace_time_ms(trace_info: TraceInfo, time_key: str) -> int:
    return trace_end_time_ms(trace_info) if time_key == "end_time_ms" else trace_info.timestamp_ms


class OnlineTraceLoader:
    def __init__(self, tracking_store: AbstractStore):
        self._tracking_store = tracking_store

    def fetch_traces(self, trace_ids: list[str]) -> list[Trace]:
        """
        Fetch full traces by their IDs.

        Attempts to load traces from the tracking store first. For any traces
        whose span data is stored in an artifact repository (e.g. S3) rather
        than in the tracking store, falls back to downloading from the artifact
        repository.

        Args:
            trace_ids: List of trace IDs to fetch.

        Returns:
            List of Trace objects (in same order as input, skipping any not found).
        """
        if not trace_ids:
            return []

        # Chunk the lookups so a very long session does not produce an oversized IN clause.
        trace_map = {}
        for i in range(0, len(trace_ids), _FETCH_CHUNK_SIZE):
            for t in self._tracking_store.batch_get_traces(trace_ids[i : i + _FETCH_CHUNK_SIZE]):
                trace_map[t.info.trace_id] = t

        # Identify traces that were not returned by batch_get_traces. This
        # happens when span data lives in the artifact repo (e.g. S3) instead
        # of the SQL tracking store.
        if missing_ids := [tid for tid in trace_ids if tid not in trace_map]:
            artifact_traces = self._fetch_traces_from_artifact_repo(missing_ids)
            trace_map.update({t.info.trace_id: t for t in artifact_traces})

        # Preserve order, skip missing
        return [trace_map[tid] for tid in trace_ids if tid in trace_map]

    def _fetch_traces_from_artifact_repo(self, trace_ids: list[str]) -> list[Trace]:
        """
        Fetch traces whose span data is stored in an artifact repository.

        Loads trace metadata from the tracking store, then downloads the span
        data from the configured artifact location (e.g. S3, GCS, ADLS).

        Args:
            trace_ids: List of trace IDs whose spans are in artifact storage.

        Returns:
            List of Trace objects with span data loaded from the artifact repo.
        """
        trace_infos = []
        for i in range(0, len(trace_ids), _FETCH_CHUNK_SIZE):
            trace_infos.extend(
                self._tracking_store.batch_get_trace_infos(trace_ids[i : i + _FETCH_CHUNK_SIZE])
            )

        traces = []
        for trace_info in trace_infos:
            if (
                trace_info.tags.get(TraceTagKey.SPANS_LOCATION)
                == SpansLocation.TRACKING_STORE.value
            ):
                # This trace should have been returned by batch_get_traces; it
                # might be partially exported. Skip it.
                _logger.debug(
                    f"Trace {trace_info.trace_id} has spans in tracking store "
                    "but was not returned by batch_get_traces, skipping"
                )
                continue

            try:
                artifact_uri = get_artifact_uri_for_trace(trace_info)
                artifact_repo = get_artifact_repository(artifact_uri)
                trace_data = TraceData.from_dict(artifact_repo.download_trace_data())
                traces.append(Trace(info=trace_info, data=trace_data))
            except Exception:
                _logger.warning(
                    f"Failed to load trace {trace_info.trace_id} from artifact repository",
                    exc_info=True,
                )

        return traces

    def fetch_trace_infos_in_range(
        self,
        experiment_id: str,
        start_time_ms: int,
        end_time_ms: int,
        filter_string: str | None = None,
        max_traces: int | None = 500,
        page_size: int = 100,
        after_trace_id: str | None = None,
        time_key: str = "timestamp_ms",
    ) -> list[TraceInfo]:
        """
        Fetch trace infos within a time window, optionally filtered.

        Args:
            experiment_id: The experiment ID to search.
            start_time_ms: Start of time window (inclusive).
            end_time_ms: End of time window (inclusive).
            filter_string: Optional additional filter criteria.
            max_traces: Maximum number of traces to return, or None to page through every
                matching trace.
            page_size: Number of traces to fetch per API call.
            after_trace_id: If set, traces at ``start_time_ms`` whose trace ID sorts at or
                before this value are skipped. This is the checkpoint tie-breaker: it lets a
                caller resume in the middle of a group of traces sharing the checkpoint
                timestamp. The exclusion is applied while paging, so ``max_traces`` counts
                only traces that are actually returned.
            time_key: Which trace time the window, ordering and tie-breaker apply to:
                ``"timestamp_ms"`` (trace start, the default) or ``"end_time_ms"`` (trace
                end, i.e. start plus duration).

        Returns:
            List of TraceInfo objects matching the criteria, ordered by
            ``(<time_key> ASC, trace_id ASC)``.
        """
        if time_key not in ("timestamp_ms", "end_time_ms"):
            raise ValueError(f"Unsupported time_key: {time_key!r}")
        time_filter = f"trace.{time_key} >= {start_time_ms} AND trace.{time_key} <= {end_time_ms}"
        combined_filter = f"{time_filter} AND {filter_string}" if filter_string else time_filter
        _logger.debug(f"Fetching traces with filter: {combined_filter}")

        all_trace_infos = []
        page_token = None

        while max_traces is None or len(all_trace_infos) < max_traces:
            batch_size = (
                page_size
                if max_traces is None
                else min(page_size, max_traces - len(all_trace_infos))
            )

            trace_batch, token = self._tracking_store.search_traces(
                experiment_ids=[experiment_id],
                filter_string=combined_filter,
                max_results=batch_size,
                order_by=[
                    f"{time_key} ASC",
                    # Order by trace ID to ensure that we have a consistent tie-breaker when
                    # multiple traces have the same timestamp and max_traces is reached in
                    # the middle of such a group
                    "request_id ASC",
                ],
                page_token=page_token,
            )

            if not trace_batch:
                break

            trace_batch = [t.info if isinstance(t, Trace) else t for t in trace_batch]
            if after_trace_id is not None:
                trace_batch = [
                    t
                    for t in trace_batch
                    if not (
                        trace_time_ms(t, time_key) == start_time_ms and t.trace_id <= after_trace_id
                    )
                ]

            if max_traces is not None:
                trace_batch = trace_batch[: max_traces - len(all_trace_infos)]
            all_trace_infos.extend(trace_batch)
            _logger.debug(
                f"Fetched batch of {len(trace_batch)} traces, total: {len(all_trace_infos)}"
            )

            if not token:
                break

            page_token = token

        _logger.debug(
            f"Fetched {len(all_trace_infos)} trace infos in range [{start_time_ms}, {end_time_ms}]"
        )

        return all_trace_infos
