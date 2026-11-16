from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities import Trace, TraceInfo
from mlflow.genai.scorers.online.trace_loader import OnlineTraceLoader
from mlflow.tracing.constant import SpansLocation, TraceTagKey


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def trace_loader(mock_store):
    return OnlineTraceLoader(mock_store)


@pytest.fixture
def sample_traces():
    traces = []
    for i in range(5):
        trace = MagicMock(spec=Trace)
        trace.info = MagicMock(spec=TraceInfo)
        trace.info.trace_id = f"trace_{i}"
        traces.append(trace)
    return traces


def test_fetch_traces_success(trace_loader, mock_store, sample_traces):
    mock_store.batch_get_traces.return_value = sample_traces
    trace_ids = [t.info.trace_id for t in sample_traces]

    result = trace_loader.fetch_traces(trace_ids)

    assert [t.info.trace_id for t in result] == trace_ids
    mock_store.batch_get_traces.assert_called_once_with(trace_ids)


def test_fetch_traces_some_missing(trace_loader, mock_store, sample_traces):
    mock_store.batch_get_traces.return_value = [
        sample_traces[0],
        sample_traces[2],
        sample_traces[4],
    ]
    # When batch_get_traces returns only some traces, the loader should try
    # fetching the rest from the artifact repo
    mock_store.batch_get_trace_infos.return_value = []
    trace_ids = [f"trace_{i}" for i in range(5)]

    result = trace_loader.fetch_traces(trace_ids)

    assert [t.info.trace_id for t in result] == ["trace_0", "trace_2", "trace_4"]
    mock_store.batch_get_trace_infos.assert_called_once_with(["trace_1", "trace_3"])


def test_fetch_traces_empty_list(trace_loader, mock_store):
    result = trace_loader.fetch_traces([])

    assert result == []
    mock_store.batch_get_traces.assert_not_called()


def test_fetch_traces_falls_back_to_artifact_repo(trace_loader, mock_store):
    mock_store.batch_get_traces.return_value = []

    trace_info = MagicMock(spec=TraceInfo)
    trace_info.trace_id = "tr-001"
    trace_info.tags = {
        TraceTagKey.SPANS_LOCATION: SpansLocation.ARTIFACT_REPO.value,
        "mlflow.artifactLocation": "s3://bucket/traces/tr-001",
    }
    mock_store.batch_get_trace_infos.return_value = [trace_info]

    mock_trace_data = {"spans": []}
    mock_artifact_repo = MagicMock()
    mock_artifact_repo.download_trace_data.return_value = mock_trace_data

    with (
        patch(
            "mlflow.genai.scorers.online.trace_loader.get_artifact_repository",
            return_value=mock_artifact_repo,
        ) as mock_get_repo,
        patch(
            "mlflow.genai.scorers.online.trace_loader.get_artifact_uri_for_trace",
            return_value="s3://bucket/traces/tr-001",
        ) as mock_get_uri,
    ):
        result = trace_loader.fetch_traces(["tr-001"])

    assert len(result) == 1
    assert result[0].info.trace_id == "tr-001"
    mock_get_uri.assert_called_once_with(trace_info)
    mock_get_repo.assert_called_once_with("s3://bucket/traces/tr-001")
    mock_artifact_repo.download_trace_data.assert_called_once()


def test_fetch_traces_mixed_storage_locations(trace_loader, mock_store):
    # trace_0 returned by batch_get_traces (in SQL store)
    sql_trace = MagicMock(spec=Trace)
    sql_trace.info = MagicMock(spec=TraceInfo)
    sql_trace.info.trace_id = "tr-sql"
    mock_store.batch_get_traces.return_value = [sql_trace]

    # trace_1 is in artifact repo
    artifact_trace_info = MagicMock(spec=TraceInfo)
    artifact_trace_info.trace_id = "tr-s3"
    artifact_trace_info.tags = {
        TraceTagKey.SPANS_LOCATION: SpansLocation.ARTIFACT_REPO.value,
        "mlflow.artifactLocation": "s3://bucket/traces/tr-s3",
    }
    mock_store.batch_get_trace_infos.return_value = [artifact_trace_info]

    mock_artifact_repo = MagicMock()
    mock_artifact_repo.download_trace_data.return_value = {"spans": []}

    with (
        patch(
            "mlflow.genai.scorers.online.trace_loader.get_artifact_repository",
            return_value=mock_artifact_repo,
        ),
        patch(
            "mlflow.genai.scorers.online.trace_loader.get_artifact_uri_for_trace",
            return_value="s3://bucket/traces/tr-s3",
        ),
    ):
        result = trace_loader.fetch_traces(["tr-sql", "tr-s3"])

    assert len(result) == 2
    assert result[0].info.trace_id == "tr-sql"
    assert result[1].info.trace_id == "tr-s3"


def test_fetch_traces_artifact_download_failure_skips_trace(trace_loader, mock_store):
    mock_store.batch_get_traces.return_value = []

    trace_info_ok = MagicMock(spec=TraceInfo)
    trace_info_ok.trace_id = "tr-ok"
    trace_info_ok.tags = {
        TraceTagKey.SPANS_LOCATION: SpansLocation.ARTIFACT_REPO.value,
        "mlflow.artifactLocation": "s3://bucket/traces/tr-ok",
    }

    trace_info_fail = MagicMock(spec=TraceInfo)
    trace_info_fail.trace_id = "tr-fail"
    trace_info_fail.tags = {
        TraceTagKey.SPANS_LOCATION: SpansLocation.ARTIFACT_REPO.value,
        "mlflow.artifactLocation": "s3://bucket/traces/tr-fail",
    }

    mock_store.batch_get_trace_infos.return_value = [trace_info_fail, trace_info_ok]

    def mock_get_uri(trace_info):
        return f"s3://bucket/traces/{trace_info.trace_id}"

    def mock_get_repo(uri):
        repo = MagicMock()
        if "tr-fail" in uri:
            repo.download_trace_data.side_effect = Exception("S3 connection failed")
        else:
            repo.download_trace_data.return_value = {"spans": []}
        return repo

    with (
        patch(
            "mlflow.genai.scorers.online.trace_loader.get_artifact_repository",
            side_effect=mock_get_repo,
        ),
        patch(
            "mlflow.genai.scorers.online.trace_loader.get_artifact_uri_for_trace",
            side_effect=mock_get_uri,
        ),
    ):
        result = trace_loader.fetch_traces(["tr-fail", "tr-ok"])

    # Only the successful trace should be returned
    assert len(result) == 1
    assert result[0].info.trace_id == "tr-ok"


def test_fetch_traces_skips_tracking_store_traces_in_fallback(trace_loader, mock_store):
    mock_store.batch_get_traces.return_value = []

    trace_info = MagicMock(spec=TraceInfo)
    trace_info.trace_id = "tr-partial"
    trace_info.tags = {
        TraceTagKey.SPANS_LOCATION: SpansLocation.TRACKING_STORE.value,
    }
    mock_store.batch_get_trace_infos.return_value = [trace_info]

    result = trace_loader.fetch_traces(["tr-partial"])

    assert len(result) == 0


def test_fetch_trace_infos_in_range_single_page(trace_loader, mock_store, sample_traces):
    mock_store.search_traces.return_value = (sample_traces, None)

    result = trace_loader.fetch_trace_infos_in_range(
        "exp1", 1000, 2000, filter_string="tags.env = 'prod'"
    )

    assert len(result) == 5
    mock_store.search_traces.assert_called_once_with(
        experiment_ids=["exp1"],
        filter_string=(
            "trace.timestamp_ms >= 1000 AND trace.timestamp_ms <= 2000 AND tags.env = 'prod'"
        ),
        max_results=100,
        order_by=["timestamp_ms ASC", "request_id ASC"],
        page_token=None,
    )


def test_fetch_trace_infos_in_range_multiple_pages(trace_loader, mock_store, sample_traces):
    mock_store.search_traces.side_effect = [
        (sample_traces[:2], "token1"),
        (sample_traces[2:4], "token2"),
        (sample_traces[4:], None),
    ]

    result = trace_loader.fetch_trace_infos_in_range("exp1", 1000, 2000, page_size=2)

    assert len(result) == 5
    assert mock_store.search_traces.call_count == 3
    calls = mock_store.search_traces.call_args_list
    assert calls[0][1]["page_token"] is None
    assert calls[1][1]["page_token"] == "token1"
    assert calls[2][1]["page_token"] == "token2"


def test_fetch_trace_infos_in_range_max_traces_limit(trace_loader, mock_store, sample_traces):
    mock_store.search_traces.return_value = (sample_traces, None)

    result = trace_loader.fetch_trace_infos_in_range("exp1", 1000, 2000, max_traces=3)

    assert len(result) == 3
    assert mock_store.search_traces.call_args[1]["max_results"] == 3


def test_fetch_trace_infos_in_range_empty_response(trace_loader, mock_store):
    mock_store.search_traces.return_value = ([], None)

    result = trace_loader.fetch_trace_infos_in_range("exp1", 1000, 2000)

    assert result == []
    mock_store.search_traces.assert_called_once()


def _info(trace_id: str, timestamp_ms: int) -> TraceInfo:
    info = MagicMock(spec=TraceInfo)
    info.trace_id = trace_id
    info.timestamp_ms = timestamp_ms
    return info


def test_fetch_trace_infos_in_range_after_trace_id_keeps_paging_past_processed_boundary(
    trace_loader, mock_store
):
    """
    Resuming from checkpoint (1000, "tr-b"): traces at 1000 with id <= "tr-b" were already
    processed. They are dropped while paging so ``max_traces`` counts only eligible traces,
    and a page consisting entirely of already-processed traces does not end discovery.
    """
    mock_store.search_traces.side_effect = [
        ([_info("tr-a", 1000), _info("tr-b", 1000)], "token1"),  # whole page already processed
        ([_info("tr-c", 1000), _info("tr-d", 1500)], "token2"),
        ([_info("tr-e", 1600)], None),
    ]

    result = trace_loader.fetch_trace_infos_in_range(
        "exp1", 1000, 2000, max_traces=3, page_size=2, after_trace_id="tr-b"
    )

    assert [t.trace_id for t in result] == ["tr-c", "tr-d", "tr-e"]
    assert mock_store.search_traces.call_count == 3


def test_fetch_trace_infos_in_range_after_trace_id_only_applies_at_window_start(
    trace_loader, mock_store
):
    # A later trace whose id sorts before the checkpoint id is still eligible.
    mock_store.search_traces.return_value = ([_info("tr-a", 1500)], None)

    result = trace_loader.fetch_trace_infos_in_range("exp1", 1000, 2000, after_trace_id="tr-z")

    assert [t.trace_id for t in result] == ["tr-a"]


def test_fetch_trace_infos_in_range_by_end_time(trace_loader, mock_store):
    """With time_key="end_time_ms" the window, ordering and checkpoint tie-break all use the
    trace end time (start + duration) so that long-running traces are found when they finish.
    """
    early_end = MagicMock(spec=TraceInfo)
    early_end.trace_id, early_end.timestamp_ms, early_end.execution_time_ms = "tr-a", 900, 100
    late_end = MagicMock(spec=TraceInfo)
    late_end.trace_id, late_end.timestamp_ms, late_end.execution_time_ms = "tr-b", 100, 1400
    mock_store.search_traces.return_value = ([early_end, late_end], None)

    result = trace_loader.fetch_trace_infos_in_range(
        "exp1", 1000, 2000, after_trace_id="tr-a", time_key="end_time_ms"
    )

    call = mock_store.search_traces.call_args[1]
    assert call["filter_string"] == "trace.end_time_ms >= 1000 AND trace.end_time_ms <= 2000"
    assert call["order_by"] == ["end_time_ms ASC", "request_id ASC"]
    # tr-a ends exactly at the window start and sorts at/before the checkpoint id: excluded.
    # tr-b started long before the window but ended inside it: included.
    assert [t.trace_id for t in result] == ["tr-b"]


def test_fetch_trace_infos_in_range_rejects_unknown_time_key(trace_loader):
    with pytest.raises(ValueError, match="Unsupported time_key"):
        trace_loader.fetch_trace_infos_in_range("exp1", 0, 1, time_key="created")


def test_fetch_trace_infos_in_range_unbounded_pages_until_exhausted(trace_loader, mock_store):
    pages = [[_info(f"tr-{i}", 1000) for i in range(100 * p, 100 * (p + 1))] for p in range(7)]
    mock_store.search_traces.side_effect = [(pg, f"tok{i}") for i, pg in enumerate(pages[:-1])] + [
        (pages[-1], None)
    ]

    result = trace_loader.fetch_trace_infos_in_range("exp1", 1000, 2000, max_traces=None)

    assert len(result) == 700
    assert mock_store.search_traces.call_count == 7
    assert all(c[1]["max_results"] == 100 for c in mock_store.search_traces.call_args_list)


def test_fetch_traces_chunks_large_id_lists(trace_loader, mock_store):
    ids = [f"tr-{i}" for i in range(450)]

    def batch_get(chunk):
        assert len(chunk) <= 200
        return [MagicMock(spec=Trace, info=MagicMock(spec=TraceInfo, trace_id=t)) for t in chunk]

    mock_store.batch_get_traces.side_effect = batch_get

    result = trace_loader.fetch_traces(ids)

    assert [t.info.trace_id for t in result] == ids
    assert mock_store.batch_get_traces.call_count == 3
