from unittest.mock import MagicMock

import pytest

from mlflow.entities import Trace, TraceInfo
from mlflow.genai.scorers.online.trace_loader import OnlineTraceLoader


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
    trace_ids = [f"trace_{i}" for i in range(5)]

    result = trace_loader.fetch_traces(trace_ids)

    assert [t.info.trace_id for t in result] == ["trace_0", "trace_2", "trace_4"]


def test_fetch_traces_empty_list(trace_loader, mock_store):
    result = trace_loader.fetch_traces([])

    assert result == []
    mock_store.batch_get_traces.assert_not_called()


def test_fetch_trace_infos_between_single_page(trace_loader, mock_store, sample_traces):
    mock_store.search_traces.return_value = (sample_traces, None)

    result = trace_loader.fetch_trace_infos_between("exp1", 1000, 2000)

    assert len(result) == 5
    mock_store.search_traces.assert_called_once()


def test_fetch_trace_infos_between_multiple_pages(trace_loader, mock_store, sample_traces):
    mock_store.search_traces.side_effect = [
        (sample_traces[:2], "token1"),
        (sample_traces[2:4], "token2"),
        (sample_traces[4:], None),
    ]

    result = trace_loader.fetch_trace_infos_between("exp1", 1000, 2000, page_size=2)

    assert len(result) == 5
    assert mock_store.search_traces.call_count == 3


def test_fetch_trace_infos_between_max_traces_limit(trace_loader, mock_store, sample_traces):
    mock_store.search_traces.return_value = (sample_traces, None)

    result = trace_loader.fetch_trace_infos_between("exp1", 1000, 2000, max_traces=3)

    assert len(result) == 3


def test_fetch_trace_infos_between_empty_response(trace_loader, mock_store):
    mock_store.search_traces.return_value = ([], None)

    result = trace_loader.fetch_trace_infos_between("exp1", 1000, 2000)

    assert result == []


def test_fetch_trace_infos_between_with_filter_string(trace_loader, mock_store, sample_traces):
    mock_store.search_traces.return_value = (sample_traces, None)

    trace_loader.fetch_trace_infos_between("exp1", 1000, 2000, filter_string="tags.env = 'prod'")

    call_kwargs = mock_store.search_traces.call_args[1]
    assert "tags.env = 'prod'" in call_kwargs["filter_string"]
    assert "trace.timestamp_ms >= 1000" in call_kwargs["filter_string"]
    assert "trace.timestamp_ms <= 2000" in call_kwargs["filter_string"]


def test_fetch_trace_infos_between_filter_string_none(trace_loader, mock_store, sample_traces):
    mock_store.search_traces.return_value = (sample_traces, None)

    trace_loader.fetch_trace_infos_between("exp1", 1000, 2000, filter_string=None)

    call_kwargs = mock_store.search_traces.call_args[1]
    assert "trace.timestamp_ms >= 1000" in call_kwargs["filter_string"]
    assert "trace.timestamp_ms <= 2000" in call_kwargs["filter_string"]
    assert (
        "AND" not in call_kwargs["filter_string"] or call_kwargs["filter_string"].count("AND") == 1
    )
