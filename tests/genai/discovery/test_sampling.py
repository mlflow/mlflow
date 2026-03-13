from unittest.mock import patch

from mlflow.genai.discovery.sampling import sample_traces
from mlflow.genai.discovery.utils import group_traces_by_session

# ---- sample_traces ----


def test_sample_traces_no_sessions(make_trace):
    traces = [make_trace() for _ in range(20)]
    search_kwargs = {"filter_string": None, "locations": ["exp-1"]}

    with patch(
        "mlflow.genai.discovery.sampling.mlflow.search_traces", return_value=traces
    ) as mock_search:
        result = sample_traces(5, search_kwargs)

    mock_search.assert_called_once()
    assert mock_search.call_args[1]["max_results"] == 25
    assert len(result) == 5
    assert all(t in traces for t in result)


def test_sample_traces_with_sessions(make_trace):
    s1_traces = [make_trace(session_id="s1") for _ in range(3)]
    s2_traces = [make_trace(session_id="s2") for _ in range(2)]
    s3_traces = [make_trace(session_id="s3") for _ in range(4)]
    all_traces = s1_traces + s2_traces + s3_traces
    search_kwargs = {"filter_string": None, "locations": ["exp-1"]}

    with patch(
        "mlflow.genai.discovery.sampling.mlflow.search_traces", return_value=all_traces
    ) as mock_search:
        result = sample_traces(2, search_kwargs)

    mock_search.assert_called_once()
    session_ids = {(t.info.trace_metadata or {}).get("mlflow.trace.session") for t in result}
    assert len(session_ids) == 2


def test_sample_traces_empty_pool():
    search_kwargs = {"filter_string": None, "locations": ["exp-1"]}

    with patch(
        "mlflow.genai.discovery.sampling.mlflow.search_traces", return_value=[]
    ) as mock_search:
        result = sample_traces(10, search_kwargs)

    mock_search.assert_called_once()
    assert result == []


def test_sample_traces_fewer_than_requested(make_trace):
    traces = [make_trace() for _ in range(3)]
    search_kwargs = {"filter_string": None, "locations": ["exp-1"]}

    with patch(
        "mlflow.genai.discovery.sampling.mlflow.search_traces", return_value=traces
    ) as mock_search:
        result = sample_traces(10, search_kwargs)

    mock_search.assert_called_once()
    assert len(result) == 3


# ---- group_traces_by_session ----


def test_group_traces_by_session_with_sessions(make_trace):
    t1 = make_trace(session_id="s1")
    t2 = make_trace(session_id="s1")
    t3 = make_trace(session_id="s2")

    groups = group_traces_by_session([t1, t2, t3])

    assert len(groups) == 2
    assert len(groups["s1"]) == 2
    assert len(groups["s2"]) == 1


def test_group_traces_by_session_no_sessions(make_trace):
    t1 = make_trace()
    t2 = make_trace()

    groups = group_traces_by_session([t1, t2])

    assert len(groups) == 2
    assert t1.info.trace_id in groups
    assert t2.info.trace_id in groups


def test_group_traces_by_session_mixed(make_trace):
    t1 = make_trace(session_id="s1")
    t2 = make_trace()

    groups = group_traces_by_session([t1, t2])

    assert len(groups) == 2
    assert len(groups["s1"]) == 1
    assert t2.info.trace_id in groups
