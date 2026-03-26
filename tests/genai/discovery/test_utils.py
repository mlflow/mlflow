import time
from unittest import mock

import pydantic
import pytest

from mlflow.entities.issue import Issue, IssueStatus
from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.discovery.constants import CATEGORY_LATENCY, build_satisfaction_instructions
from mlflow.genai.discovery.entities import _ConversationAnalysis, _IdentifiedIssue
from mlflow.genai.discovery.utils import (
    _call_llm,
    _lookup_model_cost,
    _ModelCost,
    _pydantic_to_response_format,
    _TokenCounter,
    build_summary,
    collect_affected_trace_ids,
    compute_latency_percentiles,
    format_annotation_prompt,
    format_trace_content,
    get_session_id,
    group_traces_by_session,
    log_discovery_artifacts,
)
from mlflow.genai.utils.gateway_utils import GatewayLiteLLMConfig
from mlflow.genai.utils.trace_utils import _extract_trace_timing_info
from mlflow.types.chat import ChatChoice, ChatCompletionResponse, ChatMessage, ChatUsage


def test_format_trace_content_includes_errors(make_trace):
    trace = make_trace(error_span=True)
    content = format_trace_content(trace)
    assert "Errors:" in content
    assert "Connection failed" in content


def test_format_trace_content_no_errors(make_trace):
    trace = make_trace()
    content = format_trace_content(trace)
    assert "Errors:" not in content


def test_format_trace_content_includes_timing_when_requested(make_trace):
    trace = make_trace()
    content = format_trace_content(trace, include_timing=True)
    assert "Total duration:" in content
    assert "Slowest spans:" in content


def test_format_trace_content_excludes_timing_by_default(make_trace):
    trace = make_trace()
    content = format_trace_content(trace, include_timing=False)
    assert "Total duration:" not in content
    assert "Slowest spans:" not in content


def test_extract_trace_timing_info_with_valid_trace(make_trace):
    trace = make_trace()
    timing_info = _extract_trace_timing_info(trace)
    assert timing_info is not None
    assert "duration_s" in timing_info
    assert "slowest_spans_formatted" in timing_info
    assert timing_info["duration_s"] > 0


def test_extract_trace_timing_info_returns_none_without_duration():
    trace_info = TraceInfo(
        trace_id="test",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=0,
        execution_duration=None,
        state=TraceState.OK,
    )
    trace = Trace(trace_info, TraceData())

    timing_info = _extract_trace_timing_info(trace)
    assert timing_info is None


def test_collect_affected_trace_ids_gathers_from_analyses():
    analyses = [
        _ConversationAnalysis(
            full_rationale="r1", affected_trace_ids=["t1", "t2"], execution_path="p1"
        ),
        _ConversationAnalysis(full_rationale="r2", affected_trace_ids=["t3"], execution_path="p2"),
    ]
    issue = _IdentifiedIssue(
        name="test",
        description="d",
        root_cause="rc",
        severity="high",
        example_indices=[0, 1],
        categories=[],
    )
    result = collect_affected_trace_ids(issue, analyses)
    assert result == ["t1", "t2", "t3"]


def test_collect_affected_trace_ids_skips_out_of_bounds():
    analyses = [
        _ConversationAnalysis(full_rationale="r1", affected_trace_ids=["t1"], execution_path="p1"),
    ]
    issue = _IdentifiedIssue(
        name="test",
        description="d",
        root_cause="rc",
        severity="high",
        example_indices=[0, 5],
        categories=[],
    )
    result = collect_affected_trace_ids(issue, analyses)
    assert result == ["t1"]


# ---- get_session_id ----


@pytest.mark.parametrize(
    ("session_id", "expected"),
    [
        ("sess-123", "sess-123"),
        (None, None),
    ],
)
def test_get_session_id(make_trace, session_id, expected):
    trace = make_trace(session_id=session_id)
    assert get_session_id(trace) == expected


# ---- log_discovery_artifacts ----


def test_log_discovery_artifacts_logs_text():
    mock_client = mock.MagicMock()
    with mock.patch("mlflow.MlflowClient", return_value=mock_client):
        log_discovery_artifacts("run-1", {"file.txt": "content"})
    mock_client.log_text.assert_called_once_with("run-1", "content", "file.txt")


def test_log_discovery_artifacts_skips_empty_run_id():
    with mock.patch("mlflow.MlflowClient") as mock_client_cls:
        log_discovery_artifacts("", {"file.txt": "content"})
    mock_client_cls.assert_not_called()


def test_log_discovery_artifacts_handles_exception():
    mock_client = mock.MagicMock()
    mock_client.log_text.side_effect = Exception("fail")
    with mock.patch("mlflow.MlflowClient", return_value=mock_client):
        log_discovery_artifacts("run-1", {"file.txt": "content"})
    mock_client.log_text.assert_called_once()


# ---- format_annotation_prompt ----


def test_format_annotation_prompt():
    issue = Issue(
        issue_id="i1",
        experiment_id="exp1",
        name="Test Issue",
        description="Test description",
        status=IssueStatus.PENDING,
        created_timestamp=0,
        last_updated_timestamp=0,
        root_causes=["cause1", "cause2"],
    )
    result = format_annotation_prompt(issue, "trace content here", "triage rationale here")
    assert "=== ISSUE ===" in result
    assert "Test Issue" in result
    assert "Test description" in result
    assert "cause1; cause2" in result
    assert "=== TRACE ===" in result
    assert "trace content here" in result
    assert "=== TRIAGE JUDGE RATIONALE ===" in result
    assert "triage rationale here" in result


# ---- build_summary ----


def test_build_summary_no_issues():
    result = build_summary([], 100)
    assert result == "Analyzed 100 traces. No issues found."
    assert not result.startswith("##")


def test_build_summary_with_issues():
    issues = [
        Issue(
            issue_id="i1",
            experiment_id="exp1",
            name="Slow Response",
            description="API responses are slow",
            status=IssueStatus.PENDING,
            created_timestamp=0,
            last_updated_timestamp=0,
            severity="high",
            root_causes=["Database query optimization needed"],
        ),
        Issue(
            issue_id="i2",
            experiment_id="exp1",
            name="Connection Errors",
            description="Frequent connection timeouts",
            status=IssueStatus.PENDING,
            created_timestamp=0,
            last_updated_timestamp=0,
            severity="medium",
            root_causes=["Network instability", "Upstream service issues"],
        ),
    ]
    result = build_summary(issues, 50)
    assert "Analyzed **50** traces" in result
    assert "Found **2** issues" in result
    assert not result.startswith("##")
    assert "### 1. Slow Response" in result
    assert "### 2. Connection Errors" in result
    assert "Database query optimization needed" in result
    assert "Network instability; Upstream service issues" in result


def test_token_counter_tracks_usage():
    counter = _TokenCounter()
    assert counter.input_tokens == 0
    assert counter.output_tokens == 0
    assert counter.cost_usd is None

    mock_response = mock.MagicMock()
    mock_response.usage = mock.MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response._hidden_params = {"response_cost": 0.005}

    counter.track(mock_response)

    assert counter.input_tokens == 100
    assert counter.output_tokens == 50
    assert counter.cost_usd == 0.005


def test_token_counter_tracks_gateway_response_without_hidden_params():
    counter = _TokenCounter(model="openai:/gpt-5-mini")
    response = ChatCompletionResponse(
        created=0,
        model="gpt-5-mini",
        choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content="hi"))],
        usage=ChatUsage(prompt_tokens=200, completion_tokens=80, total_tokens=280),
    )

    counter.track(response)

    assert counter.input_tokens == 200
    assert counter.output_tokens == 80
    assert counter._cost_usd == 0.0
    assert counter._model == "openai:/gpt-5-mini"


def test_token_counter_to_dict_looks_up_cost_when_zero():
    counter = _TokenCounter(input_tokens=100, output_tokens=50, model="openai:/gpt-5-mini")

    with mock.patch(
        "mlflow.genai.discovery.utils._lookup_model_cost",
        return_value=0.0042,
    ) as mock_lookup:
        result = counter.to_dict()

    mock_lookup.assert_called_once()
    assert result["cost_usd"] == 0.0042
    assert result["total_tokens"] == 150


def test_group_traces_by_session_groups_by_session_id(make_trace):
    traces = [
        make_trace(session_id="session-1"),
        make_trace(session_id="session-1"),
        make_trace(session_id="session-2"),
    ]

    result = group_traces_by_session(traces)

    assert len(result) == 2
    assert "session-1" in result
    assert "session-2" in result
    assert len(result["session-1"]) == 2
    assert len(result["session-2"]) == 1


def test_group_traces_by_session_creates_standalone_groups_for_no_session(make_trace):
    trace1 = make_trace(session_id=None)
    trace2 = make_trace(session_id=None)
    trace3 = make_trace(session_id="session-A")

    traces = [trace1, trace2, trace3]
    result = group_traces_by_session(traces)

    assert len(result) == 3
    assert trace1.info.trace_id in result
    assert trace2.info.trace_id in result
    assert "session-A" in result
    assert len(result[trace1.info.trace_id]) == 1
    assert len(result[trace2.info.trace_id]) == 1
    assert len(result["session-A"]) == 1


def test_group_traces_by_session_sorts_by_timestamp(make_trace):
    trace1 = make_trace(session_id="session-1")
    time.sleep(0.01)
    trace2 = make_trace(session_id="session-1")
    time.sleep(0.01)
    trace3 = make_trace(session_id="session-1")

    traces = [trace3, trace1, trace2]

    result = group_traces_by_session(traces)

    session_traces = result["session-1"]
    assert session_traces[0].info.trace_id == trace1.info.trace_id
    assert session_traces[1].info.trace_id == trace2.info.trace_id
    assert session_traces[2].info.trace_id == trace3.info.trace_id


def test_call_llm_uses_gateway_when_litellm_unavailable():
    with (
        mock.patch(
            "mlflow.genai.discovery.utils._is_litellm_available", return_value=False
        ) as mock_avail,
        mock.patch(
            "mlflow.genai.discovery.utils._call_llm_via_gateway",
        ) as mock_gw,
    ):
        _call_llm("openai:/gpt-5-mini", [{"role": "user", "content": "hi"}])

    mock_avail.assert_called_once()
    mock_gw.assert_called_once()


def test_call_llm_uses_litellm_when_available():
    with (
        mock.patch(
            "mlflow.genai.discovery.utils._is_litellm_available", return_value=True
        ) as mock_avail,
        mock.patch(
            "mlflow.genai.discovery.utils._call_llm_via_litellm",
        ) as mock_ll,
    ):
        _call_llm("openai:/gpt-5-mini", [{"role": "user", "content": "hi"}])

    mock_avail.assert_called_once()
    mock_ll.assert_called_once()


def test_pydantic_to_response_format():
    class MySchema(pydantic.BaseModel):
        name: str
        score: int

    result = _pydantic_to_response_format(MySchema)

    assert result["type"] == "json_schema"
    assert result["json_schema"]["name"] == "MySchema"
    schema = result["json_schema"]["schema"]
    assert "name" in schema["properties"]
    assert "score" in schema["properties"]


def test_lookup_model_cost_returns_calculated_cost():
    cost_info = _ModelCost(input_cost_per_token=0.00001, output_cost_per_token=0.00003)
    with mock.patch(
        "mlflow.genai.discovery.utils._fetch_model_cost", return_value=cost_info
    ) as mock_fetch:
        cost = _lookup_model_cost("openai:/gpt-5-mini", 1000, 500)

    mock_fetch.assert_called_once()
    assert cost == pytest.approx(1000 * 0.00001 + 500 * 0.00003)


def test_lookup_model_cost_returns_none_on_missing_model():
    with mock.patch(
        "mlflow.genai.discovery.utils._fetch_model_cost", return_value=None
    ) as mock_fetch:
        assert _lookup_model_cost("openai:/gpt-5-mini", 100, 50) is None

    mock_fetch.assert_called_once()


def test_lookup_model_cost_returns_none_on_network_error():
    with mock.patch(
        "mlflow.genai.discovery.utils._fetch_model_cost", return_value=None
    ) as mock_fetch:
        assert _lookup_model_cost("openai:/gpt-5-mini", 100, 50) is None

    mock_fetch.assert_called_once()


@pytest.mark.parametrize(
    ("durations", "expected_result"),
    [
        (
            [1000, 2000, 3000, 4000, 5000],
            {"p50": 3.0, "p75": 4.0, "p90": 4.6, "p95": 4.8, "p99": 4.96, "count": 5},
        ),
        ([], None),
        ([None, None], None),
        (
            [1000, None, 2000],
            {"p50": 1.5, "p75": 1.75, "p90": 1.9, "p95": 1.95, "p99": 1.99, "count": 2},
        ),
    ],
    ids=["valid_traces", "empty_traces", "no_durations", "skips_without_duration"],
)
def test_compute_latency_percentiles(make_trace, durations, expected_result):
    traces = [make_trace(execution_duration_ms=d) for d in durations] if durations else []
    result = compute_latency_percentiles(traces)
    assert result == expected_result


@pytest.mark.parametrize(
    ("use_conversation", "categories", "latency_stats", "expected_assertions"),
    [
        (
            False,
            [CATEGORY_LATENCY, "correctness"],
            {"p50": 1.5, "p75": 2.0, "p90": 3.0, "p95": 4.0, "count": 100},
            {
                "LATENCY CHECK:": True,
                "p50=1.5s": True,
                "p75=2.0s": True,
                "p90=3.0s": True,
                "p95=4.0s": True,
                "from 100 traces": True,
            },
        ),
        (
            False,
            [CATEGORY_LATENCY, "correctness"],
            None,
            {
                "LATENCY CHECK:": True,
                "p50=": False,
                "using this dataset's latency distribution": False,
            },
        ),
        (
            False,
            ["correctness", "relevance"],
            None,
            {"LATENCY CHECK:": False},
        ),
        (
            True,
            [CATEGORY_LATENCY],
            {"p50": 1.0, "p75": 1.5, "p90": 2.0, "p95": 2.5, "count": 50},
            {
                "LATENCY CHECK:": True,
                "p50=1.0s": True,
                "conversation": True,
            },
        ),
    ],
    ids=[
        "with_stats",
        "without_stats",
        "no_latency_category",
        "conversation_mode",
    ],
)
def test_build_satisfaction_instructions_latency_variations(
    use_conversation, categories, latency_stats, expected_assertions
):
    result = build_satisfaction_instructions(
        use_conversation=use_conversation, categories=categories, latency_stats=latency_stats
    )

    for expected_text, should_be_present in expected_assertions.items():
        if should_be_present:
            assert expected_text in result or expected_text in result.lower()
        else:
            assert expected_text not in result


def test_call_llm_handles_gateway_models():
    mock_response = mock.MagicMock()
    mock_response.usage = mock.MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    gateway_config = GatewayLiteLLMConfig(
        model="openai/test-endpoint",
        api_base="http://localhost:5000/gateway",
        api_key="test-key",
        extra_headers={"X-Custom": "header"},
    )

    with (
        mock.patch(
            "mlflow.genai.utils.gateway_utils.get_gateway_litellm_config",
            return_value=gateway_config,
        ) as mock_get_config,
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._invoke_litellm",
            return_value=mock_response,
        ) as mock_invoke,
    ):
        messages = [{"role": "user", "content": "test"}]
        result = _call_llm("gateway:/test-endpoint", messages)

        mock_get_config.assert_called_once_with("test-endpoint")
        mock_invoke.assert_called_once()

        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["litellm_model"] == "openai/test-endpoint"
        assert call_kwargs["api_base"] == "http://localhost:5000/gateway"
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["extra_headers"] == {"X-Custom": "header"}
        assert call_kwargs["messages"] == messages
        assert result == mock_response


def test_call_llm_handles_non_gateway_models():
    mock_response = mock.MagicMock()
    mock_response.usage = mock.MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    with (
        mock.patch(
            "mlflow.metrics.genai.model_utils.convert_mlflow_uri_to_litellm",
            return_value="openai/gpt-4",
        ) as mock_convert,
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._invoke_litellm",
            return_value=mock_response,
        ) as mock_invoke,
    ):
        messages = [{"role": "user", "content": "test"}]
        result = _call_llm("openai:/gpt-4", messages)

        mock_convert.assert_called_once_with("openai:/gpt-4")
        mock_invoke.assert_called_once()

        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["litellm_model"] == "openai/gpt-4"
        assert call_kwargs["api_base"] is None
        assert call_kwargs["api_key"] is None
        assert call_kwargs["extra_headers"] is None
        assert call_kwargs["messages"] == messages
        assert result == mock_response


def test_call_llm_with_json_mode():
    mock_response = mock.MagicMock()
    with mock.patch(
        "mlflow.genai.judges.adapters.litellm_adapter._invoke_litellm",
        return_value=mock_response,
    ) as mock_invoke:
        messages = [{"role": "user", "content": "test"}]
        _call_llm("openai:/gpt-4", messages, json_mode=True)

        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["include_response_format"] is True


def test_call_llm_with_response_format():
    class TestModel(pydantic.BaseModel):
        field: str

    mock_response = mock.MagicMock()
    with mock.patch(
        "mlflow.genai.judges.adapters.litellm_adapter._invoke_litellm",
        return_value=mock_response,
    ) as mock_invoke:
        messages = [{"role": "user", "content": "test"}]
        _call_llm("openai:/gpt-4", messages, response_format=TestModel)

        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["response_format"] == TestModel
        assert call_kwargs["include_response_format"] is True


def test_call_llm_tracks_tokens():
    mock_response = mock.MagicMock()
    mock_response.usage = mock.MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response._hidden_params = {"response_cost": 0.01}

    with mock.patch(
        "mlflow.genai.judges.adapters.litellm_adapter._invoke_litellm",
        return_value=mock_response,
    ):
        counter = _TokenCounter()
        messages = [{"role": "user", "content": "test"}]
        _call_llm("openai:/gpt-4", messages, token_counter=counter)

        assert counter.input_tokens == 100
        assert counter.output_tokens == 50
        assert counter.cost_usd == 0.01
