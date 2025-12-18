from unittest import mock

import pytest

from mlflow.genai import Scorer, scorer
from mlflow.genai.evaluation.telemetry import (
    _BATCH_SIZE_HEADER,
    _CLIENT_NAME_HEADER,
    _CLIENT_VERSION_HEADER,
    _SESSION_ID_HEADER,
    emit_custom_metric_event,
)
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Correctness, Guidelines
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


# Class based scorers
class IsEmpty(Scorer):
    name: str = "is_empty"

    def __call__(self, *, outputs) -> bool:
        return outputs == ""


def test_emit_custom_metric_event():
    from databricks.agents.evals import metric

    # Legacy custom metrics
    @metric
    def not_empty(response):
        return response != ""

    scorers = [
        # Built-in
        Correctness(),
        Guidelines(guidelines="The answer must be concise and straight to the point."),
        # Custom
        is_concise,
        is_correct,
        IsEmpty(),
        not_empty,
        make_judge(
            name="is_kind",
            instructions="The answer must be kind. {{ outputs }}",
            feedback_value_type=str,
        ),
    ]
    with (
        mock.patch("mlflow.genai.evaluation.telemetry.is_databricks_uri", return_value=True),
        mock.patch(
            "mlflow.genai.evaluation.telemetry.http_request", autospec=True
        ) as mock_http_request,
        mock.patch("mlflow.genai.evaluation.telemetry.get_databricks_host_creds"),
    ):
        emit_custom_metric_event(
            scorers=scorers,
            eval_count=10,
            aggregated_metrics={
                "is_concise/mean": 0.1,
                "is_concise/min": 0.2,
                "is_concise/max": 0.3,
                "is_correct/mean": 0.4,
                "is_empty/mean": 0.5,
                "not_empty/max": 0.6,
                "correctness/mean": 0.7,
                "guidelines/mean": 0.8,
                "is_kind/mean": 0.9,
            },
        )

    mock_http_request.assert_called_once()
    call_args = mock_http_request.call_args[1]

    assert call_args["method"] == "POST"
    assert call_args["endpoint"] == "/api/2.0/agents/evaluation-client-usage-events"

    headers = call_args["extra_headers"]
    assert headers[_CLIENT_VERSION_HEADER] == VERSION
    assert headers[_SESSION_ID_HEADER] is not None
    assert headers[_BATCH_SIZE_HEADER] == "10"
    assert headers[_CLIENT_NAME_HEADER] == "mlflow"

    event = call_args["json"]
    assert len(event["metric_names"]) == 5
    assert all(isinstance(name, str) for name in event["metric_names"])
    assert event["eval_count"] == 10
    assert event["metrics"] == [
        {
            "name": mock.ANY,
            "average": 0.1,
            "count": 10,
        },
        {
            "name": mock.ANY,
            "average": 0.4,
            "count": 10,
        },
        {
            "name": mock.ANY,
            "average": 0.5,
            "count": 10,
        },
        {
            "name": mock.ANY,
            "average": None,
            "count": 10,
        },
        {
            "name": mock.ANY,
            "average": 0.9,
            "count": 10,
        },
    ]
    # Metric names should be hashed
    assert isinstance(event["metrics"][0]["name"], str)
    assert event["metrics"][0]["name"] != "is_concise"


def test_emit_custom_metric_usage_event_skip_outside_databricks():
    with (
        mock.patch("mlflow.genai.evaluation.telemetry.is_databricks_uri", return_value=False),
        mock.patch(
            "mlflow.genai.evaluation.telemetry.http_request", autospec=True
        ) as mock_http_request,
        mock.patch("mlflow.genai.evaluation.telemetry.get_databricks_host_creds"),
    ):
        emit_custom_metric_event(
            scorers=[is_concise, is_correct],
            eval_count=10,
            aggregated_metrics={"is_concise/mean": 0.1, "is_correct/mean": 0.2},
        )
    mock_http_request.assert_not_called()


def test_emit_custom_metric_usage_event_with_sessions():
    with (
        mock.patch("mlflow.genai.evaluation.telemetry.is_databricks_uri", return_value=True),
        mock.patch(
            "mlflow.genai.evaluation.telemetry.http_request", autospec=True
        ) as mock_http_request,
        mock.patch("mlflow.genai.evaluation.telemetry.get_databricks_host_creds"),
    ):
        for _ in range(3):
            emit_custom_metric_event(
                scorers=[is_concise, is_correct],
                eval_count=10,
                aggregated_metrics={"is_concise/mean": 0.1, "is_correct/mean": 0.2},
            )

    assert mock_http_request.call_count == 3
    session_ids = [
        call_args[1]["extra_headers"][_SESSION_ID_HEADER]
        for call_args in mock_http_request.call_args_list
    ]
    assert len(set(session_ids)) == 1
    assert all(session_id is not None for session_id in session_ids)
