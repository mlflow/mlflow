from unittest import mock

import pytest

from mlflow.genai import Scorer, scorer
from mlflow.genai.evaluation.telemetry import (
    _BATCH_SIZE_HEADER,
    _CLIENT_NAME_HEADER,
    _CLIENT_VERSION_HEADER,
    _SESSION_ID_HEADER,
    emit_metric_usage_event,
)
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Correctness, Guidelines, UserFrustration
from mlflow.genai.scorers.validation import IS_DBX_AGENTS_INSTALLED
from mlflow.version import VERSION

if not IS_DBX_AGENTS_INSTALLED:
    pytest.skip("Skipping Databricks only test.", allow_module_level=True)


@scorer
def is_concise(outputs) -> bool:
    return len(outputs) < 100


@scorer
def is_correct(outputs, expectations) -> bool:
    return outputs == expectations["expected_response"]


class IsEmpty(Scorer):
    name: str = "is_empty"

    def __call__(self, *, outputs) -> bool:
        return outputs == ""


from databricks.agents.evals import metric


@metric
def not_empty(response):
    return response != ""


session_level_judge = make_judge(
    name="session_quality",
    instructions="Evaluate if the {{ conversation }} is coherent and complete.",
    feedback_value_type=bool,
)


@pytest.fixture
def mock_http_request():
    with (
        mock.patch("mlflow.genai.evaluation.telemetry.is_databricks_uri", return_value=True),
        mock.patch(
            "mlflow.genai.evaluation.telemetry.http_request", autospec=True
        ) as mock_http_request,
        mock.patch("mlflow.genai.evaluation.telemetry.get_databricks_host_creds"),
    ):
        yield mock_http_request


def test_emit_metric_usage_event_skip_outside_databricks():
    with (
        mock.patch("mlflow.genai.evaluation.telemetry.is_databricks_uri", return_value=False),
        mock.patch(
            "mlflow.genai.evaluation.telemetry.http_request", autospec=True
        ) as mock_http_request,
        mock.patch("mlflow.genai.evaluation.telemetry.get_databricks_host_creds"),
    ):
        emit_metric_usage_event(
            scorers=[is_concise],
            trace_count=10,
            session_count=0,
            aggregated_metrics={"is_concise/mean": 0.5},
        )
    mock_http_request.assert_not_called()


def test_emit_metric_usage_event_skip_when_no_scorers(mock_http_request):
    emit_metric_usage_event(scorers=[], trace_count=10, session_count=0, aggregated_metrics={})
    mock_http_request.assert_not_called()


def test_emit_metric_usage_event_custom_scorers_only(mock_http_request):
    is_kind = make_judge(
        name="is_kind",
        instructions="The answer must be kind. {{ outputs }}",
        feedback_value_type=str,
    )
    emit_metric_usage_event(
        scorers=[is_concise, is_correct, IsEmpty(), is_kind, not_empty],
        trace_count=10,
        session_count=0,
        aggregated_metrics={
            "is_concise/mean": 0.1,
            "is_correct/mean": 0.2,
            "is_empty/mean": 0.3,
            "is_kind/mean": 0.4,
            "not_empty/mean": 0.5,
        },
    )

    mock_http_request.assert_called_once()
    payload = mock_http_request.call_args[1]["json"]

    assert payload == {
        "agent_evaluation_client_usage_events": [
            {
                "custom_metric_usage_event": {
                    "eval_count": 10,
                    "metrics": [
                        {"name": mock.ANY, "average": 0.1, "count": 10},
                        {"name": mock.ANY, "average": 0.2, "count": 10},
                        {"name": mock.ANY, "average": 0.3, "count": 10},
                        {"name": mock.ANY, "average": 0.4, "count": 10},
                        {"name": mock.ANY, "average": 0.5, "count": 10},
                    ],
                }
            }
        ]
    }


def test_emit_metric_usage_event_builtin_scorers_only(mock_http_request):
    emit_metric_usage_event(
        scorers=[Correctness(), Guidelines(guidelines="Be concise")],
        trace_count=5,
        session_count=0,
        aggregated_metrics={"correctness/mean": 0.8, "guidelines/mean": 0.9},
    )

    mock_http_request.assert_called_once()
    payload = mock_http_request.call_args[1]["json"]

    assert payload == {
        "agent_evaluation_client_usage_events": [
            {
                "builtin_scorer_usage_event": {
                    "metrics": [
                        {"name": "Correctness", "count": 5},
                        {"name": "Guidelines", "count": 5},
                    ],
                }
            }
        ]
    }


def test_emit_metric_usage_event_mixed_custom_and_builtin_scorers(mock_http_request):
    emit_metric_usage_event(
        scorers=[Correctness(), is_concise, Guidelines(guidelines="Be concise")],
        trace_count=10,
        session_count=0,
        aggregated_metrics={
            "correctness/mean": 0.7,
            "is_concise/mean": 0.5,
            "guidelines/mean": 0.8,
        },
    )

    mock_http_request.assert_called_once()
    payload = mock_http_request.call_args[1]["json"]

    assert payload == {
        "agent_evaluation_client_usage_events": [
            {
                "custom_metric_usage_event": {
                    "eval_count": 10,
                    "metrics": [{"name": mock.ANY, "average": 0.5, "count": 10}],
                }
            },
            {
                "builtin_scorer_usage_event": {
                    "metrics": [
                        {"name": "Correctness", "count": 10},
                        {"name": "Guidelines", "count": 10},
                    ],
                }
            },
        ]
    }


def test_emit_metric_usage_event_headers(mock_http_request):
    emit_metric_usage_event(
        scorers=[is_concise],
        trace_count=10,
        session_count=0,
        aggregated_metrics={"is_concise/mean": 0.5},
    )

    call_args = mock_http_request.call_args[1]
    assert call_args["method"] == "POST"
    assert call_args["endpoint"] == "/api/2.0/agents/evaluation-client-usage-events"

    headers = call_args["extra_headers"]
    assert headers[_CLIENT_VERSION_HEADER] == VERSION
    assert headers[_SESSION_ID_HEADER] is not None
    assert headers[_BATCH_SIZE_HEADER] == "10"
    assert headers[_CLIENT_NAME_HEADER] == "mlflow"


def test_emit_metric_usage_event_with_multiple_calls(mock_http_request):
    for _ in range(3):
        emit_metric_usage_event(
            scorers=[is_concise, Correctness()],
            trace_count=10,
            session_count=0,
            aggregated_metrics={"is_concise/mean": 0.5, "correctness/mean": 0.8},
        )

    assert mock_http_request.call_count == 3
    session_ids = [
        call[1]["extra_headers"][_SESSION_ID_HEADER] for call in mock_http_request.call_args_list
    ]
    assert len(set(session_ids)) == 1


def test_emit_metric_usage_event_session_level_custom_scorer(mock_http_request):
    emit_metric_usage_event(
        scorers=[session_level_judge],
        trace_count=10,
        session_count=3,
        aggregated_metrics={"session_quality/mean": 0.7},
    )

    mock_http_request.assert_called_once()
    payload = mock_http_request.call_args[1]["json"]

    assert payload == {
        "agent_evaluation_client_usage_events": [
            {
                "custom_metric_usage_event": {
                    "eval_count": 10,
                    "metrics": [{"name": mock.ANY, "average": 0.7, "count": 3}],
                }
            }
        ]
    }


def test_emit_metric_usage_event_session_level_builtin_scorer(mock_http_request):
    emit_metric_usage_event(
        scorers=[UserFrustration()],
        trace_count=10,
        session_count=3,
        aggregated_metrics={"user_frustration/mean": 0.8},
    )

    mock_http_request.assert_called_once()
    payload = mock_http_request.call_args[1]["json"]

    assert payload == {
        "agent_evaluation_client_usage_events": [
            {
                "builtin_scorer_usage_event": {
                    "metrics": [{"name": "UserFrustration", "count": 3}],
                }
            }
        ]
    }


def test_emit_metric_usage_event_mixed_session_and_trace_level_scorers(mock_http_request):
    emit_metric_usage_event(
        scorers=[is_concise, session_level_judge, Correctness()],
        trace_count=10,
        session_count=3,
        aggregated_metrics={
            "is_concise/mean": 0.5,
            "session_quality/mean": 0.7,
            "correctness/mean": 0.8,
        },
    )

    mock_http_request.assert_called_once()
    payload = mock_http_request.call_args[1]["json"]

    assert payload == {
        "agent_evaluation_client_usage_events": [
            {
                "custom_metric_usage_event": {
                    "eval_count": 10,
                    "metrics": [
                        {"name": mock.ANY, "average": 0.5, "count": 10},
                        {"name": mock.ANY, "average": 0.7, "count": 3},
                    ],
                }
            },
            {
                "builtin_scorer_usage_event": {
                    "metrics": [{"name": "Correctness", "count": 10}],
                }
            },
        ]
    }
