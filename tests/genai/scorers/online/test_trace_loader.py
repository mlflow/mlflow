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
    """When batch_get_traces returns nothing (spans in artifact repo), the loader
    should fetch trace info and download span data from the artifact repo."""
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
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=mock_artifact_repo,
        ) as mock_get_repo,
        patch(
            "mlflow.tracing.utils.artifact_utils.get_artifact_uri_for_trace",
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
    """When some traces are in the SQL store and some in artifact storage,
    the loader should fetch from both and return all in the correct order."""
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
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=mock_artifact_repo,
        ),
        patch(
            "mlflow.tracing.utils.artifact_utils.get_artifact_uri_for_trace",
            return_value="s3://bucket/traces/tr-s3",
        ),
    ):
        result = trace_loader.fetch_traces(["tr-sql", "tr-s3"])

    assert len(result) == 2
    assert result[0].info.trace_id == "tr-sql"
    assert result[1].info.trace_id == "tr-s3"


def test_fetch_traces_artifact_download_failure_skips_trace(trace_loader, mock_store):
    """If downloading from artifact repo fails for one trace, that trace
    should be skipped and others should still be returned."""
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
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            side_effect=mock_get_repo,
        ),
        patch(
            "mlflow.tracing.utils.artifact_utils.get_artifact_uri_for_trace",
            side_effect=mock_get_uri,
        ),
    ):
        result = trace_loader.fetch_traces(["tr-fail", "tr-ok"])

    # Only the successful trace should be returned
    assert len(result) == 1
    assert result[0].info.trace_id == "tr-ok"


def test_fetch_traces_skips_tracking_store_traces_in_fallback(trace_loader, mock_store):
    """Traces marked as TRACKING_STORE that weren't returned by batch_get_traces
    (e.g. partially exported) should be skipped in the artifact fallback."""
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
