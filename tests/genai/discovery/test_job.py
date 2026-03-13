from unittest import mock

import pytest

from mlflow.entities.run_status import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.genai.discovery.job import (
    _PROVIDER_ENV_VARS,
    _fetch_provider_credentials,
    invoke_issue_detection_job,
)


@pytest.mark.parametrize(
    ("provider", "secret_value", "expected_credentials"),
    [
        ("openai", {"api_key": "test-key"}, {"OPENAI_API_KEY": "test-key"}),
        ("anthropic", {"api_key": "test-key"}, {"ANTHROPIC_API_KEY": "test-key"}),
        ("cohere", {"api_key": "test-key"}, {"COHERE_API_KEY": "test-key"}),
        ("mistral", {"api_key": "test-key"}, {"MISTRAL_API_KEY": "test-key"}),
        (
            "bedrock",
            {
                "aws_access_key_id": "access-key",
                "aws_secret_access_key": "secret-key",
                "aws_session_token": "session-token",
            },
            {
                "AWS_ACCESS_KEY_ID": "access-key",
                "AWS_SECRET_ACCESS_KEY": "secret-key",
                "AWS_SESSION_TOKEN": "session-token",
            },
        ),
        (
            "amazon-bedrock",
            {"aws_access_key_id": "access-key", "aws_secret_access_key": "secret-key"},
            {"AWS_ACCESS_KEY_ID": "access-key", "AWS_SECRET_ACCESS_KEY": "secret-key"},
        ),
    ],
)
def test_fetch_provider_credentials_success(provider, secret_value, expected_credentials):
    with mock.patch(
        "mlflow.genai.discovery.job.get_decrypted_secret", return_value=secret_value
    ) as mock_get_secret:
        credentials = _fetch_provider_credentials(provider, "secret-123")
        mock_get_secret.assert_called_once_with("secret-123")
        assert credentials == expected_credentials


def test_fetch_provider_credentials_unknown_provider():
    with pytest.raises(MlflowException, match="Unknown provider 'unknown'. Supported providers:"):
        _fetch_provider_credentials("unknown", "secret-123")


def test_fetch_provider_credentials_case_insensitive():
    with mock.patch(
        "mlflow.genai.discovery.job.get_decrypted_secret",
        return_value={"api_key": "test-key"},
    ) as mock_get_secret:
        credentials = _fetch_provider_credentials("OpenAI", "secret-123")
        mock_get_secret.assert_called_once_with("secret-123")
        assert credentials == {"OPENAI_API_KEY": "test-key"}


def test_fetch_provider_credentials_missing_api_key():
    with mock.patch(
        "mlflow.genai.discovery.job.get_decrypted_secret", return_value={}
    ) as mock_get_secret:
        credentials = _fetch_provider_credentials("openai", "secret-123")
        mock_get_secret.assert_called_once_with("secret-123")
        assert credentials == {}


def test_fetch_provider_credentials_bedrock_missing_optional_token():
    with mock.patch(
        "mlflow.genai.discovery.job.get_decrypted_secret",
        return_value={"aws_access_key_id": "access-key", "aws_secret_access_key": "secret-key"},
    ) as mock_get_secret:
        credentials = _fetch_provider_credentials("bedrock", "secret-123")
        mock_get_secret.assert_called_once_with("secret-123")
        assert credentials == {
            "AWS_ACCESS_KEY_ID": "access-key",
            "AWS_SECRET_ACCESS_KEY": "secret-key",
        }


def test_invoke_issue_detection_job_has_metadata():
    assert hasattr(invoke_issue_detection_job, "_job_fn_metadata")
    metadata = invoke_issue_detection_job._job_fn_metadata
    assert metadata.name == "invoke_issue_detection"


def test_invoke_issue_detection_job_success():
    mock_client = mock.MagicMock()
    mock_trace_1 = mock.MagicMock()
    mock_trace_2 = mock.MagicMock()
    mock_traces = [mock_trace_1, mock_trace_2]

    mock_result = mock.MagicMock()
    mock_result.summary = "Found 3 issues"
    mock_result.issues = [mock.MagicMock(), mock.MagicMock(), mock.MagicMock()]
    mock_result.total_traces_analyzed = 2
    mock_result.total_cost_usd = 0.15

    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=mock_client),
        mock.patch("mlflow.genai.discovery.job.discover_issues", return_value=mock_result),
    ):
        mock_client._tracing_client.batch_get_traces.return_value = mock_traces

        result = invoke_issue_detection_job(
            experiment_id="exp-123",
            trace_ids=["trace-1", "trace-2"],
            categories=["correctness", "safety"],
            run_id="run-123",
            model="openai:/gpt-4o",
        )

        mock_client.link_traces_to_run.assert_called_once_with(["trace-1", "trace-2"], "run-123")
        mock_client._tracing_client.batch_get_traces.assert_called_once_with(["trace-1", "trace-2"])
        mock_client.set_terminated.assert_called_once_with(
            "run-123", RunStatus.to_string(RunStatus.FINISHED)
        )
        mock_client.set_tag.assert_called_once_with("run-123", "total_cost_usd", 0.15)

        assert result["summary"] == "Found 3 issues"
        assert result["issues"] == 3
        assert result["total_traces_analyzed"] == 2
        assert result["total_cost_usd"] == 0.15


def test_invoke_issue_detection_job_with_endpoint():
    mock_client = mock.MagicMock()
    mock_traces = [mock.MagicMock()]

    mock_result = mock.MagicMock()
    mock_result.summary = "Found 1 issue"
    mock_result.issues = [mock.MagicMock()]
    mock_result.total_traces_analyzed = 1
    mock_result.total_cost_usd = 0.05

    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=mock_client),
        mock.patch(
            "mlflow.genai.discovery.job.discover_issues", return_value=mock_result
        ) as mock_discover_issues,
    ):
        mock_client._tracing_client.batch_get_traces.return_value = mock_traces

        invoke_issue_detection_job(
            experiment_id="exp-123",
            trace_ids=["trace-1"],
            categories=["correctness"],
            run_id="run-123",
            model="gateway:/my-endpoint",
        )

        # Verify gateway endpoint format is passed to discover_issues
        mock_discover_issues.assert_called_once()
        call_args = mock_discover_issues.call_args
        assert call_args.kwargs["model"] == "gateway:/my-endpoint"


def test_invoke_issue_detection_job_batches_large_trace_list():
    mock_client = mock.MagicMock()
    mock_traces = [mock.MagicMock() for _ in range(250)]

    mock_result = mock.MagicMock()
    mock_result.summary = "Summary"
    mock_result.issues = []
    mock_result.total_traces_analyzed = 250
    mock_result.total_cost_usd = 1.0

    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=mock_client),
        mock.patch("mlflow.genai.discovery.job.discover_issues", return_value=mock_result),
    ):
        mock_client._tracing_client.batch_get_traces.return_value = mock_traces

        trace_ids = [f"trace-{i}" for i in range(250)]
        invoke_issue_detection_job(
            experiment_id="exp-123",
            trace_ids=trace_ids,
            categories=["correctness"],
            run_id="run-123",
            model="openai:/gpt-4o",
        )

        # Should batch link_traces_to_run calls (100 traces per call)
        assert mock_client.link_traces_to_run.call_count == 3
        call_args_list = mock_client.link_traces_to_run.call_args_list
        assert len(call_args_list[0][0][0]) == 100
        assert len(call_args_list[1][0][0]) == 100
        assert len(call_args_list[2][0][0]) == 50


def test_invoke_issue_detection_job_failure_marks_run_failed():
    mock_client = mock.MagicMock()
    mock_client._tracing_client.batch_get_traces.side_effect = Exception("API error")

    with mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=mock_client):
        with pytest.raises(Exception, match="API error"):
            invoke_issue_detection_job(
                experiment_id="exp-123",
                trace_ids=["trace-1"],
                categories=["correctness"],
                run_id="run-123",
                model="openai:/gpt-4o",
            )

        mock_client.set_terminated.assert_called_once_with(
            "run-123", RunStatus.to_string(RunStatus.FAILED)
        )


def test_provider_env_vars_contains_expected_providers():
    assert "openai" in _PROVIDER_ENV_VARS
    assert "anthropic" in _PROVIDER_ENV_VARS
    assert "bedrock" in _PROVIDER_ENV_VARS
    assert "amazon-bedrock" in _PROVIDER_ENV_VARS
    assert "cohere" in _PROVIDER_ENV_VARS
    assert "mistral" in _PROVIDER_ENV_VARS
    assert _PROVIDER_ENV_VARS["openai"] == "OPENAI_API_KEY"
    assert _PROVIDER_ENV_VARS["anthropic"] == "ANTHROPIC_API_KEY"
    assert isinstance(_PROVIDER_ENV_VARS["bedrock"], dict)
