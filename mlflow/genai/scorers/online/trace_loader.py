"""Trace loading utilities for online scoring."""

import logging

from mlflow.entities import Trace, TraceData, TraceInfo
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import SpansLocation, TraceTagKey
from mlflow.tracing.utils.artifact_utils import get_artifact_uri_for_trace

_logger = logging.getLogger(__name__)


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

        traces = self._tracking_store.batch_get_traces(trace_ids)
        trace_map = {t.info.trace_id: t for t in traces}

        # Identify traces that were not returned by batch_get_traces. This
        # happens when span data lives in the artifact repo (e.g. S3) instead
        # of the SQL tracking store.
        missing_ids = [tid for tid in trace_ids if tid not in trace_map]
        if missing_ids:
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
        trace_infos = self._tracking_store.batch_get_trace_infos(trace_ids)

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
        max_traces: int = 500,
        page_size: int = 100,
    ) -> list[TraceInfo]:
        """
        Fetch trace infos within a time window, optionally filtered.

        Args:
            experiment_id: The experiment ID to search.
            start_time_ms: Start of time window (inclusive).
            end_time_ms: End of time window (inclusive).
            filter_string: Optional additional filter criteria.
            max_traces: Maximum number of traces to return.
            page_size: Number of traces to fetch per API call.

        Returns:
            List of TraceInfo objects matching the criteria.
        """
        time_filter = (
            f"trace.timestamp_ms >= {start_time_ms} AND trace.timestamp_ms <= {end_time_ms}"
        )
        combined_filter = f"{time_filter} AND {filter_string}" if filter_string else time_filter
        _logger.debug(f"Fetching traces with filter: {combined_filter}")

        all_trace_infos = []
        page_token = None

        while len(all_trace_infos) < max_traces:
            batch_size = min(page_size, max_traces - len(all_trace_infos))

            trace_batch, token = self._tracking_store.search_traces(
                experiment_ids=[experiment_id],
                filter_string=combined_filter,
                max_results=batch_size,
                order_by=[
                    "timestamp_ms ASC",
                    # Order by trace ID to ensure that we have a consistent tie-breaker when
                    # multiple traces have the same timestamp and max_traces is reached in
                    # the middle of such a group
                    "request_id ASC",
                ],
                page_token=page_token,
            )

            if not trace_batch:
                break

            remaining = max_traces - len(all_trace_infos)
            all_trace_infos.extend(trace_batch[:remaining])
            _logger.debug(
                f"Fetched batch of {len(trace_batch)} traces, total: {len(all_trace_infos)}"
            )

            if not token:
                break

            page_token = token

        _logger.debug(
            f"Fetched {len(all_trace_infos)} trace infos in range [{start_time_ms}, {end_time_ms}]"
        )

        return [t.info if isinstance(t, Trace) else t for t in all_trace_infos]
