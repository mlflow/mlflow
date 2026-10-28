from unittest import mock

import pytest

import mlflow
from mlflow.entities.run_status import RunStatus
from mlflow.exceptions import MlflowException
from mlflow.genai.discovery.job import (
    _fetch_provider_credentials,
    invoke_issue_detection_job,
    run_scheduled_issue_detection_job,
)
from mlflow.server import handlers
from mlflow.server.constants import ARTIFACT_ROOT_ENV_VAR, BACKEND_STORE_URI_ENV_VAR


@pytest.mark.parametrize(
    ("provider", "secret_value", "auth_config", "expected_credentials"),
    [
        ("openai", {"api_key": "test-key"}, {}, {"OPENAI_API_KEY": "test-key"}),
        (
            "openai",
            {"api_key": "test-key"},
            {"api_base": "https://custom.example.com/v1"},
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_API_BASE": "https://custom.example.com/v1",
            },
        ),
        ("anthropic", {"api_key": "test-key"}, {}, {"ANTHROPIC_API_KEY": "test-key"}),
        ("gemini", {"api_key": "test-key"}, {}, {"GEMINI_API_KEY": "test-key"}),
        (
            "azure",
            {"api_key": "test-key"},
            {"api_base": "https://my-resource.openai.azure.com", "api_version": "2024-02-01"},
            {
                "AZURE_API_KEY": "test-key",
                "AZURE_API_BASE": "https://my-resource.openai.azure.com",
                "AZURE_API_VERSION": "2024-02-01",
            },
        ),
        (
            "bedrock",
            {
                "aws_access_key_id": "access-key",
                "aws_secret_access_key": "secret-key",
                "aws_session_token": "session-token",
            },
            {},
            {
                "AWS_ACCESS_KEY_ID": "access-key",
                "AWS_SECRET_ACCESS_KEY": "secret-key",
                "AWS_SESSION_TOKEN": "session-token",
            },
        ),
    ],
)
def test_fetch_provider_credentials_success(
    provider, secret_value, auth_config, expected_credentials
):
    mock_store = mock.MagicMock()
    mock_store._get_decrypted_secret.return_value = secret_value
    mock_store.get_secret_info.return_value.auth_config = auth_config
    credentials = _fetch_provider_credentials(mock_store, provider, "secret-123")
    mock_store._get_decrypted_secret.assert_called_once_with("secret-123")
    mock_store.get_secret_info.assert_called_once_with(secret_id="secret-123")
    assert credentials == expected_credentials


def test_fetch_provider_credentials_unknown_provider():
    mock_store = mock.MagicMock()
    with pytest.raises(MlflowException, match="Unknown provider 'unknown'. Supported providers:"):
        _fetch_provider_credentials(mock_store, "unknown", "secret-123")


def test_fetch_provider_credentials_case_insensitive():
    mock_store = mock.MagicMock()
    mock_store._get_decrypted_secret.return_value = {"api_key": "test-key"}
    mock_store.get_secret_info.return_value.auth_config = {}
    credentials = _fetch_provider_credentials(mock_store, "OpenAI", "secret-123")
    mock_store._get_decrypted_secret.assert_called_once_with("secret-123")
    mock_store.get_secret_info.assert_called_once_with(secret_id="secret-123")
    assert credentials == {"OPENAI_API_KEY": "test-key"}


def test_fetch_provider_credentials_missing_api_key():
    mock_store = mock.MagicMock()
    mock_store._get_decrypted_secret.return_value = {}
    mock_store.get_secret_info.return_value.auth_config = {}
    credentials = _fetch_provider_credentials(mock_store, "openai", "secret-123")
    mock_store._get_decrypted_secret.assert_called_once_with("secret-123")
    mock_store.get_secret_info.assert_called_once_with(secret_id="secret-123")
    assert credentials == {}


def test_fetch_provider_credentials_bedrock_missing_optional_token():
    mock_store = mock.MagicMock()
    mock_store._get_decrypted_secret.return_value = {
        "aws_access_key_id": "access-key",
        "aws_secret_access_key": "secret-key",
    }
    mock_store.get_secret_info.return_value.auth_config = {}
    credentials = _fetch_provider_credentials(mock_store, "bedrock", "secret-123")
    mock_store._get_decrypted_secret.assert_called_once_with("secret-123")
    mock_store.get_secret_info.assert_called_once_with(secret_id="secret-123")
    assert credentials == {
        "AWS_ACCESS_KEY_ID": "access-key",
        "AWS_SECRET_ACCESS_KEY": "secret-key",
    }


def test_fetch_provider_credentials_none_auth_config():
    mock_store = mock.MagicMock()
    mock_store._get_decrypted_secret.return_value = {"api_key": "test-key"}
    mock_store.get_secret_info.return_value.auth_config = None
    credentials = _fetch_provider_credentials(mock_store, "openai", "secret-123")
    mock_store._get_decrypted_secret.assert_called_once_with("secret-123")
    mock_store.get_secret_info.assert_called_once_with(secret_id="secret-123")
    assert credentials == {"OPENAI_API_KEY": "test-key"}


def test_fetch_provider_credentials_azure_partial_auth_config():
    mock_store = mock.MagicMock()
    mock_store._get_decrypted_secret.return_value = {"api_key": "test-key"}
    mock_store.get_secret_info.return_value.auth_config = {
        "api_base": "https://my-resource.openai.azure.com"
    }
    credentials = _fetch_provider_credentials(mock_store, "azure", "secret-123")
    mock_store._get_decrypted_secret.assert_called_once_with("secret-123")
    mock_store.get_secret_info.assert_called_once_with(secret_id="secret-123")
    assert credentials == {
        "AZURE_API_KEY": "test-key",
        "AZURE_API_BASE": "https://my-resource.openai.azure.com",
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


def test_invoke_issue_detection_job_forwards_the_saved_priorities():
    # The settings the experiment carries only mean something if they reach the pipeline, and
    # the job is the one hop between the handler and `discover_issues` that can drop them.
    mock_client = mock.MagicMock()
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=mock_client),
        mock.patch("mlflow.genai.discovery.job.discover_issues") as discover,
    ):
        mock_client._tracing_client.batch_get_traces.return_value = [mock.MagicMock()]
        invoke_issue_detection_job(
            experiment_id="exp-123",
            trace_ids=["trace-1"],
            categories=["correctness"],
            run_id="run-123",
            model="openai:/gpt-4o",
            instructions="Only report high severity issues.",
        )

    assert discover.call_args.kwargs["instructions"] == "Only report high severity issues."


def test_invoke_issue_detection_job_without_priorities_passes_none():
    mock_client = mock.MagicMock()
    with (
        mock.patch("mlflow.genai.discovery.job.MlflowClient", return_value=mock_client),
        mock.patch("mlflow.genai.discovery.job.discover_issues") as discover,
    ):
        mock_client._tracing_client.batch_get_traces.return_value = [mock.MagicMock()]
        invoke_issue_detection_job(
            experiment_id="exp-123",
            trace_ids=["trace-1"],
            categories=["correctness"],
            run_id="run-123",
        )

    assert discover.call_args.kwargs["instructions"] is None


def _store_backed_tracking(monkeypatch, tmp_path, name):
    """Stand in for the job runner's environment: tracking points at the server, the store is set.

    Returns the store URI the job is expected to end up talking to.
    """
    store_uri = f"sqlite:///{tmp_path / name}"
    monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, store_uri)
    monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, str(tmp_path / "artifacts"))
    monkeypatch.setattr(handlers, "_tracking_store", None)
    mlflow.set_tracking_uri("http://127.0.0.1:1")
    return store_uri


def test_invoke_issue_detection_job_builds_its_client_against_the_store(monkeypatch, tmp_path):
    """
    A job cannot authenticate to the tracking API, so a client built against the server's HTTP
    URI fails with 401 on an auth-enabled server. The client this job builds must see the store.
    """
    previous = mlflow.get_tracking_uri()
    store_uri = _store_backed_tracking(monkeypatch, tmp_path, "detect-store.db")
    seen: dict[str, str] = {}

    def _build_client():
        seen["tracking_uri"] = mlflow.get_tracking_uri()
        return mock.MagicMock()

    try:
        with (
            mock.patch("mlflow.genai.discovery.job.MlflowClient", side_effect=_build_client),
            mock.patch("mlflow.genai.discovery.job.discover_issues"),
        ):
            invoke_issue_detection_job(
                experiment_id="exp-123",
                trace_ids=["trace-1"],
                categories=["correctness"],
                run_id="run-123",
            )
    finally:
        mlflow.set_tracking_uri(previous)

    assert seen["tracking_uri"] == store_uri


def test_scheduled_issue_detection_job_builds_its_client_against_the_store(monkeypatch, tmp_path):
    previous = mlflow.get_tracking_uri()
    store_uri = _store_backed_tracking(monkeypatch, tmp_path, "scheduled-store.db")
    seen: dict[str, str] = {}

    def _build_client():
        seen["tracking_uri"] = mlflow.get_tracking_uri()
        raise RuntimeError("stop after the client is built")

    try:
        with mock.patch("mlflow.genai.discovery.job.MlflowClient", side_effect=_build_client):
            with pytest.raises(RuntimeError, match="stop after the client is built"):
                run_scheduled_issue_detection_job(
                    experiment_id="exp-123",
                    window_lower_bound_ms=0,
                    window_upper_bound_ms=1,
                    categories=["correctness"],
                )
    finally:
        mlflow.set_tracking_uri(previous)

    assert seen["tracking_uri"] == store_uri
